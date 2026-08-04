

import os
import math
import numpy as np
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
#  SETTINGS — tuned for LFW-127 (small dataset, 127 subjects)
# ============================================================
DATA_DIR      = r"C:\Users\khanm\Desktop\UESTC\PhD. Paper\5th_paper_IEEE_TIFS\Feauture_extraction\LFW127_outlier_removed"
NUM_SUBJECTS  = 127
NUM_SAMPLES   = 12
TRAIN_SAMPLES = [1, 2, 3, 4, 5, 6, 7, 8]      # 8 train samples
TEST_SAMPLES  = [9, 10, 11, 12]                # 4 test samples (more stable EER with small subject count)
FEATURE_DIM   = 512
PROTECT_DIM   = 512
NUM_KEYS      = 10
KEY_DIM       = 64

EPOCHS        = 100       # small dataset converges fast, but give enough epochs for warmup+cosine schedule
BATCH_SIZE    = 32        # smaller batch since only 127*8=1016 training samples total
LR_MAIN       = 3e-4
LR_CRITIC     = 1e-4
LR_ADV        = 1e-4

# Security loss weights — same warmup approach as CASIA-WebFace v2
DELTA_PER_DIM   = 0.3
DELTA           = DELTA_PER_DIM * PROTECT_DIM
GAMMA_FINAL     = 0.25
LAMBDA_FINAL    = 0.25
ALPHA_FINAL     = 0.15
MU_FINAL        = 0.25
WARMUP_EPOCHS   = 15

ARC_SCALE     = 64.0
ARC_MARGIN    = 0.30

EER_STEP      = 0.001
EVAL_EVERY    = 10
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
#  REPRODUCIBILITY — fix all sources of randomness
# ============================================================
SEED = 42

import random

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # for multi-GPU, harmless otherwise
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Force deterministic algorithms (may slightly slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)

set_seed(SEED)

print(f"Device: {DEVICE}  |  Feature dim: {FEATURE_DIM}")
print(f"Subjects: {NUM_SUBJECTS}  |  Train samples: {TRAIN_SAMPLES}  |  Test samples: {TEST_SAMPLES}")
print(f"Epochs: {EPOCHS}  |  Warmup: {WARMUP_EPOCHS}  |  Batch: {BATCH_SIZE}")


# ============================================================
#  LOAD DATA — separate train and test, reads .npy files
# ============================================================
def load_lfw127_features(data_dir):
    train_raw = {}
    test_raw  = {}
    for i in range(1, NUM_SUBJECTS + 1):
        for j in range(1, NUM_SAMPLES + 1):
            fp = os.path.join(data_dir, f"{i}_{j}.npy")
            if not os.path.exists(fp):
                continue
            A = np.load(fp).astype(np.float32).flatten()
            if j in TRAIN_SAMPLES:
                train_raw[(i, j)] = A
            elif j in TEST_SAMPLES:
                test_raw[(i, j)]  = A
    print(f"Train templates: {len(train_raw)}  |  Test templates: {len(test_raw)}")
    return train_raw, test_raw


# ============================================================
#  DATASET
# ============================================================
class FaceDataset(Dataset):
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
#  NETWORKS — same architecture as CASIA-WebFace v2 (proven)
# ============================================================
def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class G_phi(nn.Module):
    """
    Section 3.1: Key-conditioned stochastic generator with residual connection.
    mu = residual_proj(x) + mu_head(h)  -- preserves identity signal directly.
    """
    def __init__(self):
        super().__init__()
        self.key_emb = nn.Embedding(NUM_KEYS, KEY_DIM)
        self.shared  = nn.Sequential(
            nn.Linear(FEATURE_DIM + KEY_DIM, 1024), nn.LayerNorm(1024), nn.ReLU(),
            nn.Linear(1024, 512),                   nn.LayerNorm(512),  nn.ReLU(),
        )
        self.mu_head    = nn.Linear(512, PROTECT_DIM)
        self.sigma_head = nn.Sequential(nn.Linear(512, PROTECT_DIM), nn.Softplus())
        self.residual_proj = nn.Linear(FEATURE_DIM, PROTECT_DIM)

        self.apply(_init)
        with torch.no_grad():
            if FEATURE_DIM == PROTECT_DIM:
                self.residual_proj.weight.copy_(torch.eye(PROTECT_DIM))
                self.residual_proj.bias.zero_()

    def forward(self, x, key, deterministic=False):
        h        = self.shared(torch.cat([x, self.key_emb(key)], dim=-1))
        residual = self.residual_proj(x)
        mu       = residual + self.mu_head(h)
        sigma    = torch.clamp(self.sigma_head(h), 1e-4, 10.0)
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
        x_norm = F.normalize(x,          dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        cosine = x_norm @ w_norm.T
        sine   = torch.sqrt(1.0 - cosine.pow(2) + 1e-8)
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = F.one_hot(label, NUM_SUBJECTS).float()
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return F.cross_entropy(output * self.s, label)


class CriticNet(nn.Module):
    """Section 3.2: MINE critic T_psi."""
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
    """Section 3.2: CLUB network q_xi."""
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
    """Section 3.3: Key-mismatched adversary A_eta."""
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
#  EER HELPERS
# ============================================================
def cos_sim(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0

def genuine_scores(prot):
    g = []
    for i in range(1, NUM_SUBJECTS+1):
        av = [j for j in TEST_SAMPLES if (i,j) in prot]
        for j1, j2 in combinations(av, 2):
            g.append(cos_sim(prot[(i,j1)], prot[(i,j2)]))
    return np.array(g)

def impostor_scores_full(prot):
    imp = []
    for s1, s2 in combinations(range(1, NUM_SUBJECTS+1), 2):
        a1 = [j for j in TEST_SAMPLES if (s1,j) in prot]
        a2 = [j for j in TEST_SAMPLES if (s2,j) in prot]
        for j1 in a1:
            for j2 in a2:
                imp.append(cos_sim(prot[(s1,j1)], prot[(s2,j2)]))
    return np.array(imp)

def compute_eer_quick(gen, imp):
    gen = gen[~np.isnan(gen)]; imp = imp[~np.isnan(imp)]
    if len(gen) == 0 or len(imp) == 0:
        return None
    start = min(gen.min(), imp.min())
    stop  = max(gen.max(), imp.max())
    ths = np.arange(start, stop+EER_STEP, EER_STEP)
    FAR, FRR = [], []
    for t in ths:
        FAR.append(np.mean(imp>=t))
        FRR.append(1-np.mean(gen>=t))
    FAR=np.array(FAR); FRR=np.array(FRR)
    ind = np.argmin(np.abs(FAR-FRR))
    return (FAR[ind]+FRR[ind])/2

def evaluate_test_eer(g_phi, test_raw, fixed_key=0):
    """Full test EER (only 127 subjects, so no need to subsample for speed)."""
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
    imp = impostor_scores_full(prot)
    eer = compute_eer_quick(gen, imp)
    g_phi.train()
    return eer


def get_warmup_weight(epoch, warmup_epochs, final_weight):
    if epoch >= warmup_epochs:
        return final_weight
    return final_weight * (epoch / warmup_epochs)


# ============================================================
#  TRAINING
# ============================================================
def train(train_raw, test_raw):
    print("\n=== Training IT-ACB (LFW-127) ===")
    loader = DataLoader(FaceDataset(train_raw), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True)

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
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=EPOCHS)

    best_eer   = 1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        g_phi.train(); adv.train(); critic.train()
        t_rec = t_sec = t_adv = t_sep = 0.0

        gamma_e  = get_warmup_weight(epoch, WARMUP_EPOCHS, GAMMA_FINAL)
        lambda_e = get_warmup_weight(epoch, WARMUP_EPOCHS, LAMBDA_FINAL)
        alpha_e  = get_warmup_weight(epoch, WARMUP_EPOCHS, ALPHA_FINAL)
        mu_e     = get_warmup_weight(epoch, WARMUP_EPOCHS, MU_FINAL)

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
            mi       = mine_read(critic, Z_K, x)
            y_oh     = F.one_hot(y, NUM_SUBJECTS).float()
            L_cl     = club_loss(club, Z_K, Z_K2, y_oh)
            L_sec    = mi + gamma_e * L_cl
            L_sep    = sep_loss_fn(mu1, s1, mu2, s2)
            L_adv    = adv_outer_loss_for_generator(adv, Z_K, x, key)

            loss = L_rec + lambda_e * L_sec + alpha_e * L_adv + mu_e * L_sep

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

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{EPOCHS}] "
                  f"Rec={t_rec/n:.3f}  Sec={t_sec/n:.3f}  "
                  f"Adv={t_adv/n:.3f}  Sep={t_sep/n:.4f}  "
                  f"w(g,l,a,m)=({gamma_e:.2f},{lambda_e:.2f},"
                  f"{alpha_e:.2f},{mu_e:.2f})  "
                  f"LR={scheduler.get_last_lr()[0]:.1e}", end="")

            if epoch % EVAL_EVERY == 0 or epoch == 1:
                eer = evaluate_test_eer(g_phi, test_raw)
                if eer is not None:
                    print(f"  |  TEST EER={eer*100:.2f}%", end="")
                    if eer < best_eer:
                        best_eer   = eer
                        best_state = {k: v.cpu().clone()
                                     for k, v in g_phi.state_dict().items()}
                        print("  (best so far)", end="")
            print()

    print(f"\nTraining complete. Best TEST EER during training: {best_eer*100:.2f}%")

    if best_state is not None:
        g_phi.load_state_dict(best_state)
        g_phi.to(DEVICE)
        print("Restored best model checkpoint.")

    return g_phi


# ============================================================
#  EXTRACT + FINAL EVAL
# ============================================================
def extract(g_phi, raw, fixed_key=0):
    g_phi.eval()
    out = {}
    with torch.no_grad():
        for (s, j), feat in raw.items():
            fn  = feat / (np.linalg.norm(feat) + 1e-8)
            x   = torch.tensor(fn).unsqueeze(0).to(DEVICE)
            key = torch.tensor([fixed_key]).to(DEVICE)
            mu, _, _ = g_phi(x, key, deterministic=True)
            mu_norm  = F.normalize(mu, dim=-1)
            z_np     = mu_norm.squeeze(0).cpu().numpy()
            if not np.isnan(z_np).any():
                out[(s, j)] = z_np
    print(f"Templates extracted: {len(out)}")
    return out


def compute_eer_full(gen, imp):
    gen = gen[~np.isnan(gen)]; imp = imp[~np.isnan(imp)]
    start = min(gen.min(), imp.min())
    stop  = max(gen.max(), imp.max())
    print(f"Threshold: {start:.4f} to {stop:.4f}")
    ths = np.arange(start, stop+EER_STEP, EER_STEP)
    FAR, FRR, GAR, TSR = [], [], [], []
    for t in ths:
        gar=np.mean(gen>=t); frr=1-gar
        far=np.mean(imp>=t); tsr=1-far
        GAR.append(gar); FRR.append(frr); FAR.append(far); TSR.append(tsr)
    FAR=np.array(FAR); FRR=np.array(FRR); GAR=np.array(GAR); TSR=np.array(TSR)
    ind = np.argmin(np.abs(FAR-FRR))
    EER = (FAR[ind]+FRR[ind])/2
    print("\n-------------------------------------")
    print("      Performance Result (IT-ACB, LFW-127)")
    print("-------------------------------------")
    print(f"Verification Rate      : {TSR[ind]*100:.2f} %")
    print(f"Genuine Acceptance Rate: {GAR[ind]*100:.2f} %")
    print(f"False Acceptance Rate  : {FAR[ind]*100:.2f} %")
    print(f"False Rejection Rate   : {FRR[ind]*100:.2f} %")
    print(f"Equal Error Rate (EER) : {EER*100:.2f} %")
    print("-------------------------------------")
    return EER, FAR, FRR, GAR


def plot_roc(FAR, GAR, eer):
    plt.figure(figsize=(6,5))
    plt.plot(FAR*100, GAR*100, 'b-', lw=2, label='IT-ACB (LFW-127)')
    plt.scatter([eer*100],[(1-eer)*100], c='red', zorder=5,
                label=f'EER={eer*100:.2f}%')
    plt.xlabel('FAR (%)'); plt.ylabel('GAR (%)')
    plt.title('ROC — LFW-127 (IT-ACB)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('roc_IT_ACB_lfw127.png', dpi=150)
    print("ROC saved: roc_IT_ACB_lfw127.png")


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    print("=== Step 1: Load LFW-127 (cleaned, train/test split) ===")
    train_raw, test_raw = load_lfw127_features(DATA_DIR)

    print("\n=== Step 2: Train IT-ACB ===")
    g_phi = train(train_raw, test_raw)

    print("\n=== Step 3: Extract FULL TEST templates ===")
    prot = extract(g_phi, test_raw)

    print("\n=== Step 4: Compute FULL scores ===")
    gen = genuine_scores(prot)
    imp = impostor_scores_full(prot)

    print("\n=== Step 5: Compute FINAL EER ===")
    EER, FAR, FRR, GAR = compute_eer_full(gen, imp)
    plot_roc(FAR, GAR, EER)
    print(f"\nFinal EER: {EER*100:.2f}%")

    torch.save(g_phi.state_dict(), 'g_phi_lfw127.pth')
    print("Model saved: g_phi_lfw127.pth")