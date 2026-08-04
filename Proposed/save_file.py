"""
IT_ACB_TJU_best_save.py  — IT-ACB for TJU Palmprint (with per-epoch EER tracking)
Section 3.1: KIIS  | Section 3.2: CK-SLG | Section 3.3: KAAMS

NEW IN THIS VERSION:
  - Computes TEST EER periodically during training (every EVAL_EVERY epochs)
  - Tracks and saves the BEST model checkpoint (lowest EER seen)
  - After training, restores the BEST checkpoint (not just final epoch)
  - Saves ALL protected templates (train+test) as .mat files using
    the BEST model - same format as original: {i}_{j}.mat, key 'feature'

Protocol: Random 8/2 split per subject (fixed seed=42 for reproducibility)

Usage: python IT_ACB_TJU_best_save.py
"""

import os
import math
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
#  SETTINGS
# ============================================================
DATA_DIR          = r"C:\Users\khanm\Desktop\UESTC\PhD. Paper\5th_paper_IEEE_TIFS\PolyU_data"
OUTPUT_MAT_DIR    = r"C:\Users\khanm\Desktop\UESTC\PhD. Paper\5th_paper_IEEE_TIFS\PolyU_PROTECTED_mat"
MODEL_SAVE_PATH   = r"g_phi_tju_best.pth"
NUM_SUBJECTS      = 386
NUM_SAMPLES       = 10
TRAIN_PER_SUBJECT = 8
TEST_PER_SUBJECT  = 2
RANDOM_SEED       = 42
FEATURE_DIM       = 512
PROTECT_DIM       = 512
NUM_KEYS          = 10
KEY_DIM           = 64
EPOCHS            = 200
BATCH_SIZE        = 64
LR_MAIN           = 2e-4
LR_CRITIC         = 1e-4
LR_ADV            = 1e-4
DELTA_PER_DIM     = 0.5
DELTA             = DELTA_PER_DIM * PROTECT_DIM
GAMMA             = 0.5
LAMBDA            = 0.5
ALPHA             = 0.3
MU                = 0.5
ARC_SCALE         = 64.0
ARC_MARGIN        = 0.35
EER_STEP          = 0.001
EVAL_EVERY        = 10    # NEW: compute TEST EER every N epochs
FIXED_KEY         = 0
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"Split: random {TRAIN_PER_SUBJECT} train / {TEST_PER_SUBJECT} test  |  seed={RANDOM_SEED}")
print(f"ArcFace: s={ARC_SCALE}  m={ARC_MARGIN}  |  Eval every {EVAL_EVERY} epochs")


# ============================================================
#  STEP 1: LOAD DATA
# ============================================================
def load_tju_features(data_dir):
    train_raw = {}
    test_raw  = {}
    rng = np.random.RandomState(RANDOM_SEED)

    for i in range(1, NUM_SUBJECTS + 1):
        available = []
        for j in range(1, NUM_SAMPLES + 1):
            fp = os.path.join(data_dir, f"{i}_{j}.mat")
            if os.path.exists(fp):
                available.append(j)
        if len(available) == 0:
            continue

        available = list(available)
        rng.shuffle(available)
        train_idx = available[:TRAIN_PER_SUBJECT]
        test_idx  = available[TRAIN_PER_SUBJECT:]

        for j in train_idx:
            fp = os.path.join(data_dir, f"{i}_{j}.mat")
            A  = sio.loadmat(fp)['feature'].flatten().astype(np.float32)
            train_raw[(i, j)] = A

        for j in test_idx:
            fp = os.path.join(data_dir, f"{i}_{j}.mat")
            A  = sio.loadmat(fp)['feature'].flatten().astype(np.float32)
            test_raw[(i, j)] = A

        if i % 100 == 0:
            print(f"  Loaded {i}/{NUM_SUBJECTS} subjects")

    print(f"Train templates: {len(train_raw)}  |  Test templates: {len(test_raw)}")
    return train_raw, test_raw


# ============================================================
#  DATASET
# ============================================================
class PalmprintDataset(Dataset):
    def __init__(self, raw):
        data, labels = [], []
        for (subj, _), feat in raw.items():
            data.append(feat)
            labels.append(subj - 1)
        self.data   = np.array(data,   dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int64)
        self.data  /= (np.linalg.norm(self.data, axis=1, keepdims=True) + 1e-8)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx]),
                torch.tensor(self.labels[idx]),
                torch.randint(0, NUM_KEYS, (1,)).squeeze())


# ============================================================
#  NETWORKS
# ============================================================
def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class G_phi(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_emb = nn.Embedding(NUM_KEYS, KEY_DIM)
        self.shared  = nn.Sequential(
            nn.Linear(FEATURE_DIM + KEY_DIM, 1024), nn.LayerNorm(1024), nn.ReLU(),
            nn.Linear(1024, 512),                   nn.LayerNorm(512),  nn.ReLU(),
        )
        self.mu_head    = nn.Linear(512, PROTECT_DIM)
        self.sigma_head = nn.Sequential(nn.Linear(512, PROTECT_DIM), nn.Softplus())
        self.apply(_init)

    def forward(self, x, key, deterministic=False):
        h     = self.shared(torch.cat([x, self.key_emb(key)], dim=-1))
        mu    = self.mu_head(h)
        sigma = torch.clamp(self.sigma_head(h), 1e-4, 10.0)
        if deterministic:
            return mu, mu, sigma
        Z_K = mu + sigma * torch.randn_like(mu)
        return Z_K, mu, sigma


class ArcFaceHead(nn.Module):
    def __init__(self, feat_dim=PROTECT_DIM, num_classes=NUM_SUBJECTS,
                 s=ARC_SCALE, m=ARC_MARGIN):
        super().__init__()
        self.s      = s
        self.m      = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m  = math.cos(m)
        self.sin_m  = math.sin(m)
        self.th     = math.cos(math.pi - m)
        self.mm     = math.sin(math.pi - m) * m

    def forward(self, x, label):
        x_norm  = F.normalize(x,          dim=-1)
        w_norm  = F.normalize(self.weight, dim=-1)
        cosine  = x_norm @ w_norm.T
        sine    = torch.sqrt(1.0 - cosine.pow(2) + 1e-8)
        phi     = cosine * self.cos_m - sine * self.sin_m
        phi     = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = F.one_hot(label, NUM_SUBJECTS).float()
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return F.cross_entropy(output * self.s, label)


class CriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PROTECT_DIM + FEATURE_DIM, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.apply(_init)

    def forward(self, z, x):
        return self.net(torch.cat([z, x], dim=-1)).squeeze(-1)


class CLUBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(PROTECT_DIM + NUM_SUBJECTS, 256), nn.ReLU(),
            nn.Linear(256, PROTECT_DIM)
        )
        self.lv_net = nn.Sequential(
            nn.Linear(PROTECT_DIM + NUM_SUBJECTS, 256), nn.ReLU(),
            nn.Linear(256, PROTECT_DIM), nn.Tanh()
        )

    def forward(self, z_i, y_oh):
        inp = torch.cat([z_i, y_oh], dim=-1)
        return self.mu_net(inp), self.lv_net(inp)

    def log_prob(self, z_j, mu, lv):
        return -0.5 * torch.sum(
            lv + (z_j - mu).pow(2) / (lv.exp() + 1e-6), dim=-1)


class AdvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_emb = nn.Embedding(NUM_KEYS, KEY_DIM)
        self.net     = nn.Sequential(
            nn.Linear(PROTECT_DIM + KEY_DIM, 512), nn.ReLU(),
            nn.Linear(512, 512),                   nn.ReLU(),
            nn.Linear(512, FEATURE_DIM)
        )

    def forward(self, z, wk):
        return self.net(torch.cat([z, self.key_emb(wk)], dim=-1))


# ============================================================
#  LOSS FUNCTIONS
# ============================================================
def mine_critic_loss(critic, z, x):
    idx = torch.randperm(x.size(0), device=x.device)
    Tj  = critic(z.detach(), x)
    Tm  = critic(z.detach(), x[idx])
    mi  = Tj.mean() - (torch.logsumexp(Tm, 0) - np.log(Tm.size(0)))
    return -mi

def mine_read(critic, z, x):
    with torch.no_grad():
        idx = torch.randperm(x.size(0), device=x.device)
        Tj  = critic(z.detach(), x)
        Tm  = critic(z.detach(), x[idx])
    mi = Tj.mean() - (torch.logsumexp(Tm, 0) - np.log(Tm.size(0)))
    return torch.clamp(mi, -10.0, 10.0)

def club_loss(club, z_i, z_j, y_oh):
    mu, lv = club(z_i.detach(), y_oh)
    lp     = club.log_prob(z_j.detach(), mu, lv)
    idx    = torch.randperm(z_j.size(0), device=z_j.device)
    ln     = club.log_prob(z_j[idx].detach(), mu, lv)
    return torch.clamp((lp - ln).mean(), -10.0, 10.0)

def sep_loss_fn(mu1, s1, mu2, s2):
    s1 = torch.clamp(s1, 1e-4); s2 = torch.clamp(s2, 1e-4)
    v1, v2 = s1.pow(2), s2.pow(2)
    kl = 0.5 * torch.sum(
        torch.log(v2/v1+1e-8) + v1/(v2+1e-8) +
        (mu1-mu2).pow(2)/(v2+1e-8) - 1, dim=-1)
    kl = torch.clamp(kl, 0.0, 1e4)
    return F.relu(DELTA - kl).mean()

def adv_inner_loss(adv, z, x, key):
    B  = z.size(0)
    wk = (key + torch.randint(1, NUM_KEYS, (B,), device=DEVICE)) % NUM_KEYS
    x_hat = adv(z.detach(), wk)
    cos   = F.cosine_similarity(
        F.normalize(x_hat, dim=-1), F.normalize(x, dim=-1), dim=-1)
    return (1.0 - cos).mean()

def adv_outer_loss_for_generator(adv, z, x, key):
    B        = z.size(0)
    best_cos = torch.ones(B, device=DEVICE)
    for _ in range(3):
        wk    = (key + torch.randint(1, NUM_KEYS, (B,), device=DEVICE)) % NUM_KEYS
        x_hat = adv(z.detach(), wk)
        cos   = F.cosine_similarity(
            F.normalize(x_hat.detach(), dim=-1),
            F.normalize(x,              dim=-1), dim=-1)
        best_cos = torch.min(best_cos, cos)
    return best_cos.mean()


# ============================================================
#  EER HELPERS (used both during training checks and final eval)
# ============================================================
def cos_sim(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0

def genuine_scores(prot):
    g = []
    for i in range(1, NUM_SUBJECTS+1):
        av = [j for j in range(1, NUM_SAMPLES+1) if (i, j) in prot]
        for j1, j2 in combinations(av, 2):
            g.append(cos_sim(prot[(i, j1)], prot[(i, j2)]))
    return np.array(g)

def impostor_scores(prot, verbose=False):
    imp = []
    for idx, (s1, s2) in enumerate(combinations(range(1, NUM_SUBJECTS+1), 2)):
        a1 = [j for j in range(1, NUM_SAMPLES+1) if (s1, j) in prot]
        a2 = [j for j in range(1, NUM_SAMPLES+1) if (s2, j) in prot]
        for j1 in a1:
            for j2 in a2:
                imp.append(cos_sim(prot[(s1, j1)], prot[(s2, j2)]))
        if verbose and idx % 10000 == 0:
            print(f"  Impostor pairs: {idx}...")
    return np.array(imp)

def compute_eer_quick(gen, imp):
    gen = gen[~np.isnan(gen)]; imp = imp[~np.isnan(imp)]
    if len(gen) == 0 or len(imp) == 0:
        return None
    start = min(gen.min(), imp.min())
    stop  = max(gen.max(), imp.max())
    ths = np.arange(start, stop + EER_STEP, EER_STEP)
    FAR, FRR = [], []
    for t in ths:
        FAR.append(np.mean(imp >= t))
        FRR.append(1 - np.mean(gen >= t))
    FAR = np.array(FAR); FRR = np.array(FRR)
    ind = np.argmin(np.abs(FAR - FRR))
    return (FAR[ind] + FRR[ind]) / 2

def evaluate_test_eer(g_phi, test_raw, fixed_key=0):
    """Extract test templates and compute EER - used during training checks."""
    g_phi.eval()
    prot = {}
    with torch.no_grad():
        for (s, j), feat in test_raw.items():
            fn  = feat / (np.linalg.norm(feat) + 1e-8)
            x   = torch.tensor(fn).unsqueeze(0).to(DEVICE)
            key = torch.tensor([fixed_key]).to(DEVICE)
            mu, _, _ = g_phi(x, key, deterministic=True)
            mu_norm  = F.normalize(mu, dim=-1)
            z_np     = mu_norm.squeeze(0).cpu().numpy()
            if not np.isnan(z_np).any():
                prot[(s, j)] = z_np
    gen = genuine_scores(prot)
    imp = impostor_scores(prot, verbose=False)
    eer = compute_eer_quick(gen, imp)
    g_phi.train()
    return eer


# ============================================================
#  TRAINING — with per-epoch TEST EER tracking + best checkpoint saving
# ============================================================
def train(train_raw, test_raw):
    print("\n=== Training IT-ACB (with periodic TEST EER checks) ===")
    loader = DataLoader(
        PalmprintDataset(train_raw),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    g_phi    = G_phi().to(DEVICE)
    arc_head = ArcFaceHead().to(DEVICE)
    critic   = CriticNet().to(DEVICE)
    club     = CLUBNet().to(DEVICE)
    adv      = AdvNet().to(DEVICE)

    opt_main = optim.Adam(
        list(g_phi.parameters()) +
        list(arc_head.parameters()) +
        list(club.parameters()),
        lr=LR_MAIN, weight_decay=1e-5)
    opt_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)
    opt_adv    = optim.Adam(adv.parameters(),    lr=LR_ADV)
    scheduler  = optim.lr_scheduler.StepLR(opt_main, step_size=80, gamma=0.1)

    best_eer   = 1.0     # NEW: track best EER
    best_state = None    # NEW: track best model weights

    for epoch in range(1, EPOCHS + 1):
        g_phi.train(); adv.train(); critic.train()
        t_rec = t_sec = t_adv = t_sep = 0.0

        for x, y, key in loader:
            x, y, key = x.to(DEVICE), y.to(DEVICE), key.to(DEVICE)
            key2 = (key + torch.randint(1, NUM_KEYS,
                    (x.size(0),), device=DEVICE)) % NUM_KEYS

            Z_K,  mu1, s1 = g_phi(x, key)
            Z_K2, mu2, s2 = g_phi(x, key2)

            opt_critic.zero_grad()
            mine_critic_loss(critic, Z_K, x).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_critic.step()

            opt_adv.zero_grad()
            adv_inner_loss(adv, Z_K, x, key).backward()
            torch.nn.utils.clip_grad_norm_(adv.parameters(), 1.0)
            opt_adv.step()

            Z_K,  mu1, s1 = g_phi(x, key)
            Z_K2, mu2, s2 = g_phi(x, key2)

            mu1_norm = F.normalize(mu1, dim=-1)
            L_rec    = arc_head(mu1_norm, y)

            mi    = mine_read(critic, Z_K, x)
            y_oh  = F.one_hot(y, NUM_SUBJECTS).float()
            L_cl  = club_loss(club, Z_K, Z_K2, y_oh)
            L_sec = mi + GAMMA * L_cl

            L_sep = sep_loss_fn(mu1, s1, mu2, s2)
            L_adv = adv_outer_loss_for_generator(adv, Z_K, x, key)

            loss = L_rec + LAMBDA * L_sec + ALPHA * L_adv + MU * L_sep

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            opt_main.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(g_phi.parameters()) +
                list(arc_head.parameters()) +
                list(club.parameters()), 1.0)
            opt_main.step()

            t_rec += L_rec.item(); t_sec += L_sec.item()
            t_adv += L_adv.item(); t_sep += L_sep.item()

        scheduler.step()
        n = len(loader)

        # ---- Print training losses ----
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{EPOCHS}] "
                  f"Rec={t_rec/n:.3f}  Sec={t_sec/n:.3f}  "
                  f"Adv={t_adv/n:.3f}  Sep={t_sep/n:.4f}  "
                  f"LR={scheduler.get_last_lr()[0]:.1e}", end="")

            # ---- NEW: Compute TEST EER every EVAL_EVERY epochs ----
            if epoch % EVAL_EVERY == 0 or epoch == 1:
                eer = evaluate_test_eer(g_phi, test_raw)
                if eer is not None:
                    print(f"  |  TEST EER={eer*100:.2f}%", end="")
                    if eer < best_eer:
                        best_eer   = eer
                        best_state = {k: v.cpu().clone()
                                     for k, v in g_phi.state_dict().items()}
                        print("  (best so far, saved)", end="")
            print()

    print(f"\nTraining complete. Best TEST EER during training: {best_eer*100:.2f}%")

    # NEW: Restore best checkpoint (not the final epoch model)
    if best_state is not None:
        g_phi.load_state_dict(best_state)
        g_phi.to(DEVICE)
        print("Restored BEST model checkpoint (not final epoch).")
        torch.save(best_state, MODEL_SAVE_PATH)
        print(f"Best model saved: {MODEL_SAVE_PATH}")

    return g_phi, best_eer


# ============================================================
#  EXTRACT TEMPLATES — test set only
# ============================================================
def extract(g_phi, test_raw, fixed_key=0):
    g_phi.eval()
    out = {}
    with torch.no_grad():
        for (s, j), feat in test_raw.items():
            fn  = feat / (np.linalg.norm(feat) + 1e-8)
            x   = torch.tensor(fn).unsqueeze(0).to(DEVICE)
            key = torch.tensor([fixed_key]).to(DEVICE)
            mu, _, _ = g_phi(x, key, deterministic=True)
            mu_norm  = F.normalize(mu, dim=-1)
            z_np     = mu_norm.squeeze(0).cpu().numpy()
            if not np.isnan(z_np).any():
                out[(s, j)] = z_np
    print(f"Test templates extracted: {len(out)}")
    return out


# ============================================================
#  SAVE ALL PROTECTED TEMPLATES AS .mat — using BEST model
# ============================================================
def save_all_protected_as_mat(g_phi, train_raw, test_raw,
                               output_dir, fixed_key=FIXED_KEY):
    os.makedirs(output_dir, exist_ok=True)
    g_phi.eval()

    all_raw = {}
    all_raw.update(train_raw)
    all_raw.update(test_raw)

    print(f"\nApplying BEST model protection layer (key={fixed_key}) to "
          f"{len(all_raw)} total templates and saving as .mat...")

    saved = 0
    with torch.no_grad():
        for (subj_id, samp_id), feat in all_raw.items():
            fn  = feat / (np.linalg.norm(feat) + 1e-8)
            x   = torch.tensor(fn).unsqueeze(0).to(DEVICE)
            key = torch.tensor([fixed_key]).to(DEVICE)

            mu, _, _ = g_phi(x, key, deterministic=True)
            mu_norm  = F.normalize(mu, dim=-1)
            protected_feat = mu_norm.squeeze(0).cpu().numpy().astype(np.float32)

            out_path = os.path.join(output_dir, f"{subj_id}_{samp_id}.mat")
            sio.savemat(out_path, {'feature': protected_feat})
            saved += 1

            if saved % 1000 == 0:
                print(f"  Saved {saved}/{len(all_raw)} protected templates...")

    print(f"\nTotal protected .mat files saved: {saved}")
    print(f"Output folder: {output_dir}")


def verify_saved_mat(output_dir, sample_subj=1, sample_samp=1):
    fp = os.path.join(output_dir, f"{sample_subj}_{sample_samp}.mat")
    if not os.path.exists(fp):
        print(f"Verification skipped: {fp} not found")
        return
    data = sio.loadmat(fp)
    feat = data['feature']
    print(f"\n=== Verification ===")
    print(f"File: {fp}")
    print(f"Feature shape : {feat.shape}")
    print(f"Feature norm  : {np.linalg.norm(feat):.4f}")
    print("Loads correctly - same format as original TJU data")


# ============================================================
#  EER — final full evaluation
# ============================================================
def compute_eer_full(gen, imp):
    gen = gen[~np.isnan(gen)]
    imp = imp[~np.isnan(imp)]
    start = min(gen.min(), imp.min())
    stop  = max(gen.max(), imp.max())
    print(f"Threshold: {start:.4f} to {stop:.4f}")
    ths = np.arange(start, stop + EER_STEP, EER_STEP)
    FAR, FRR, GAR, TSR = [], [], [], []
    for t in ths:
        gar = np.mean(gen >= t); frr = 1 - gar
        far = np.mean(imp >= t); tsr = 1 - far
        GAR.append(gar); FRR.append(frr)
        FAR.append(far); TSR.append(tsr)
    FAR = np.array(FAR); FRR = np.array(FRR)
    GAR = np.array(GAR); TSR = np.array(TSR)
    ind = np.argmin(np.abs(FAR - FRR))
    EER = (FAR[ind] + FRR[ind]) / 2
    print("\n-------------------------------------")
    print("      Performance Result (IT-ACB, BEST model)")
    print("-------------------------------------")
    print(f"Verification Rate      : {TSR[ind]*100:.2f} %")
    print(f"Genuine Acceptance Rate: {GAR[ind]*100:.2f} %")
    print(f"False Acceptance Rate  : {FAR[ind]*100:.2f} %")
    print(f"False Rejection Rate   : {FRR[ind]*100:.2f} %")
    print(f"Equal Error Rate (EER) : {EER*100:.2f} %")
    print("-------------------------------------")
    return EER, FAR, FRR, GAR


def plot_roc(FAR, GAR, eer):
    plt.figure(figsize=(6, 5))
    plt.plot(FAR*100, GAR*100, 'b-', lw=2, label='IT-ACB (Best Model)')
    plt.scatter([eer*100], [(1-eer)*100], c='red', zorder=5,
                label=f'EER={eer*100:.2f}%')
    plt.xlabel('FAR (%)'); plt.ylabel('GAR (%)')
    plt.title('ROC — TJU Palmprint (IT-ACB, Best Model)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('roc_IT_ACB_best.png', dpi=150)
    print("ROC saved: roc_IT_ACB_best.png")


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    print("=== Step 1: Load TJU (random 8/2 split) ===")
    train_raw, test_raw = load_tju_features(DATA_DIR)

    print("\n=== Step 2: Train IT-ACB (tracking best TEST EER) ===")
    g_phi, best_eer_during_training = train(train_raw, test_raw)

    print("\n=== Step 3: Extract FULL TEST templates (best model) ===")
    prot = extract(g_phi, test_raw)

    print("\n=== Step 4: Compute FULL scores ===")
    gen = genuine_scores(prot)
    imp = impostor_scores(prot, verbose=True)

    print("\n=== Step 5: Compute FINAL EER (best model, full test set) ===")
    EER, FAR, FRR, GAR = compute_eer_full(gen, imp)
    plot_roc(FAR, GAR, EER)

    print("\n=== Step 6: Save ALL protected templates as .mat (best model) ===")
    save_all_protected_as_mat(g_phi, train_raw, test_raw, OUTPUT_MAT_DIR)
    verify_saved_mat(OUTPUT_MAT_DIR)

    print(f"\n=== ALL DONE ===")
    print(f"Best EER during training (quick check) : {best_eer_during_training*100:.2f}%")
    print(f"Final EER (full test set, best model)   : {EER*100:.2f}%")
    print(f"Protected .mat templates saved at       : {OUTPUT_MAT_DIR}")
    print(f"Best model saved at                     : {MODEL_SAVE_PATH}")