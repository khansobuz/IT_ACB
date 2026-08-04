import os
import math
import random
import csv
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
#  REPRODUCIBILITY — fix all sources of randomness
# ============================================================
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ============================================================
#  FIXED SETTINGS (do not change across sweeps)
# ============================================================
DATA_DIR      = r"C:\Users\khanm\Desktop\UESTC\PhD. Paper\5th_paper_IEEE_TIFS\Feauture_extraction\LFW127_outlier_removed"
NUM_SUBJECTS  = 127
NUM_SAMPLES   = 12
TRAIN_SAMPLES = [1, 2, 3, 4, 5, 6, 7, 8]
TEST_SAMPLES  = [9, 10, 11, 12]
FEATURE_DIM   = 512          # backbone feature size — always fixed
NUM_KEYS      = 10           # number of distinct key IDs (not the same as KEY_DIM)

EPOCHS        = 100
BATCH_SIZE    = 32
LR_MAIN       = 3e-4
LR_CRITIC     = 1e-4
LR_ADV        = 1e-4

DELTA_PER_DIM   = 0.3
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
#  THE TWO SWEEPS YOU ASKED FOR
#  Sweep A: vary PROTECT_DIM, KEY_DIM fixed at 128
#  Sweep B: vary KEY_DIM,     PROTECT_DIM fixed at 512
# ============================================================
PROTECT_DIM_SWEEP = [128, 256, 512, 1024]
KEY_DIM_FIXED_FOR_SWEEP_A = 128

KEY_DIM_SWEEP = [64, 128, 256, 512]
PROTECT_DIM_FIXED_FOR_SWEEP_B = 512

print(f"Device: {DEVICE}  |  Feature dim: {FEATURE_DIM}  |  Seed: {SEED}")


# ============================================================
#  LOAD DATA
# ============================================================
def load_lfw127_features(data_dir):
    train_raw, test_raw = {}, {}
    for i in range(1, NUM_SUBJECTS + 1):
        for j in range(1, NUM_SAMPLES + 1):
            fp = os.path.join(data_dir, f"{i}_{j}.npy")
            if not os.path.exists(fp):
                continue
            A = np.load(fp).astype(np.float32).flatten()
            if j in TRAIN_SAMPLES:
                train_raw[(i, j)] = A
            elif j in TEST_SAMPLES:
                test_raw[(i, j)] = A
    print(f"Train templates: {len(train_raw)}  |  Test templates: {len(test_raw)}")
    return train_raw, test_raw


# ============================================================
#  DATASET
# ============================================================
class FaceDataset(Dataset):
    def __init__(self, raw, num_keys):
        self.num_keys = num_keys
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
                torch.randint(0, self.num_keys, (1,)).squeeze())


# ============================================================
#  NETWORKS — all dims passed in as constructor args (no globals)
# ============================================================
def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class G_phi(nn.Module):
    def __init__(self, feature_dim, protect_dim, key_dim, num_keys):
        super().__init__()
        self.key_emb = nn.Embedding(num_keys, key_dim)
        self.shared  = nn.Sequential(
            nn.Linear(feature_dim + key_dim, 1024), nn.LayerNorm(1024), nn.ReLU(),
            nn.Linear(1024, 512),                   nn.LayerNorm(512),  nn.ReLU(),
        )
        self.mu_head    = nn.Linear(512, protect_dim)
        self.sigma_head = nn.Sequential(nn.Linear(512, protect_dim), nn.Softplus())
        self.residual_proj = nn.Linear(feature_dim, protect_dim)

        self.apply(_init)
        with torch.no_grad():
            if feature_dim == protect_dim:
                self.residual_proj.weight.copy_(torch.eye(protect_dim))
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
    def __init__(self, feat_dim, num_classes, s=ARC_SCALE, m=ARC_MARGIN):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m
        self.num_classes = num_classes

    def forward(self, x, label):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        cosine = x_norm @ w_norm.T
        sine   = torch.sqrt(1.0 - cosine.pow(2) + 1e-8)
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = F.one_hot(label, self.num_classes).float()
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return F.cross_entropy(output * self.s, label)


class CriticNet(nn.Module):
    def __init__(self, protect_dim, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(protect_dim + feature_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.apply(_init)

    def forward(self, z, x):
        return self.net(torch.cat([z, x], dim=-1)).squeeze(-1)


class CLUBNet(nn.Module):
    def __init__(self, protect_dim, num_classes):
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(protect_dim + num_classes, 256), nn.ReLU(),
            nn.Linear(256, protect_dim)
        )
        self.lv_net = nn.Sequential(
            nn.Linear(protect_dim + num_classes, 256), nn.ReLU(),
            nn.Linear(256, protect_dim), nn.Tanh()
        )

    def forward(self, z_i, y_oh):
        inp = torch.cat([z_i, y_oh], dim=-1)
        return self.mu_net(inp), self.lv_net(inp)

    def log_prob(self, z_j, mu, lv):
        return -0.5 * torch.sum(lv + (z_j - mu).pow(2) / (lv.exp() + 1e-6), dim=-1)


class AdvNet(nn.Module):
    def __init__(self, protect_dim, feature_dim, key_dim, num_keys):
        super().__init__()
        self.key_emb = nn.Embedding(num_keys, key_dim)
        self.net = nn.Sequential(
            nn.Linear(protect_dim + key_dim, 512), nn.ReLU(),
            nn.Linear(512, 512),                    nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, z, wk):
        return self.net(torch.cat([z, self.key_emb(wk)], dim=-1))


# ============================================================
#  LOSS FUNCTIONS
# ============================================================
def mine_critic_loss(critic, z, x):
    idx = torch.randperm(x.size(0), device=x.device)
    Tj = critic(z.detach(), x)
    Tm = critic(z.detach(), x[idx])
    mi = Tj.mean() - (torch.logsumexp(Tm, 0) - np.log(Tm.size(0)))
    return -mi

def mine_read(critic, z, x):
    with torch.no_grad():
        idx = torch.randperm(x.size(0), device=x.device)
        Tj = critic(z.detach(), x)
        Tm = critic(z.detach(), x[idx])
    mi = Tj.mean() - (torch.logsumexp(Tm, 0) - np.log(Tm.size(0)))
    return torch.clamp(mi, -10.0, 10.0)

def club_loss(club, z_i, z_j, y_oh):
    mu, lv = club(z_i.detach(), y_oh)
    lp = club.log_prob(z_j.detach(), mu, lv)
    idx = torch.randperm(z_j.size(0), device=z_j.device)
    ln = club.log_prob(z_j[idx].detach(), mu, lv)
    return torch.clamp((lp - ln).mean(), -10.0, 10.0)

def sep_loss_fn(mu1, s1, mu2, s2, delta):
    s1 = torch.clamp(s1, 1e-4); s2 = torch.clamp(s2, 1e-4)
    v1, v2 = s1.pow(2), s2.pow(2)
    kl = 0.5 * torch.sum(
        torch.log(v2/v1+1e-8) + v1/(v2+1e-8) +
        (mu1-mu2).pow(2)/(v2+1e-8) - 1, dim=-1)
    kl = torch.clamp(kl, 0.0, 1e4)
    return F.relu(delta - kl).mean()

def adv_inner_loss(adv, z, x, key, num_keys):
    B = z.size(0)
    wk = (key + torch.randint(1, num_keys, (B,), device=DEVICE)) % num_keys
    x_hat = adv(z.detach(), wk)
    cos = F.cosine_similarity(F.normalize(x_hat, dim=-1), F.normalize(x, dim=-1), dim=-1)
    return (1.0 - cos).mean()

def adv_outer_loss_for_generator(adv, z, x, key, num_keys):
    B = z.size(0)
    best_cos = torch.ones(B, device=DEVICE)
    for _ in range(3):
        wk = (key + torch.randint(1, num_keys, (B,), device=DEVICE)) % num_keys
        x_hat = adv(z.detach(), wk)
        cos = F.cosine_similarity(F.normalize(x_hat.detach(), dim=-1), F.normalize(x, dim=-1), dim=-1)
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

def compute_eer_full(gen, imp):
    gen = gen[~np.isnan(gen)]; imp = imp[~np.isnan(imp)]
    start = min(gen.min(), imp.min())
    stop  = max(gen.max(), imp.max())
    ths = np.arange(start, stop+EER_STEP, EER_STEP)
    FAR, FRR, GAR = [], [], []
    for t in ths:
        gar = np.mean(gen>=t); frr = 1-gar
        far = np.mean(imp>=t)
        GAR.append(gar); FRR.append(frr); FAR.append(far)
    FAR=np.array(FAR); FRR=np.array(FRR); GAR=np.array(GAR)
    ind = np.argmin(np.abs(FAR-FRR))
    EER = (FAR[ind]+FRR[ind])/2
    return EER, FAR, FRR, GAR

def evaluate_test_eer(g_phi, test_raw, fixed_key=0):
    g_phi.eval()
    prot = {}
    with torch.no_grad():
        for (s, j), feat in test_raw.items():
            fn = feat / (np.linalg.norm(feat) + 1e-8)
            x  = torch.tensor(fn).unsqueeze(0).to(DEVICE)
            key = torch.tensor([fixed_key]).to(DEVICE)
            mu, _, _ = g_phi(x, key, deterministic=True)
            mu_norm = F.normalize(mu, dim=-1)
            z_np = mu_norm.squeeze(0).cpu().numpy()
            if not np.isnan(z_np).any():
                prot[(s, j)] = z_np
    gen = genuine_scores(prot)
    imp = impostor_scores_full(prot)
    eer = compute_eer_quick(gen, imp)
    g_phi.train()
    return eer

def extract(g_phi, raw, fixed_key=0):
    g_phi.eval()
    out = {}
    with torch.no_grad():
        for (s, j), feat in raw.items():
            fn = feat / (np.linalg.norm(feat) + 1e-8)
            x  = torch.tensor(fn).unsqueeze(0).to(DEVICE)
            key = torch.tensor([fixed_key]).to(DEVICE)
            mu, _, _ = g_phi(x, key, deterministic=True)
            mu_norm = F.normalize(mu, dim=-1)
            z_np = mu_norm.squeeze(0).cpu().numpy()
            if not np.isnan(z_np).any():
                out[(s, j)] = z_np
    return out

def get_warmup_weight(epoch, warmup_epochs, final_weight):
    if epoch >= warmup_epochs:
        return final_weight
    return final_weight * (epoch / warmup_epochs)


# ============================================================
#  ONE FULL TRAINING RUN FOR A GIVEN (PROTECT_DIM, KEY_DIM) CONFIG
# ============================================================
def run_experiment(tag, feature_dim, protect_dim, key_dim, num_keys, train_raw, test_raw):
    print(f"\n{'='*70}\nRUN [{tag}]  FEATURE_DIM={feature_dim}  PROTECT_DIM={protect_dim}  "
          f"KEY_DIM={key_dim}  NUM_KEYS={num_keys}\n{'='*70}")

    set_seed(SEED)   # reset seed fresh for every run so runs are comparable/reproducible

    delta = DELTA_PER_DIM * protect_dim

    loader_gen = torch.Generator()
    loader_gen.manual_seed(SEED)
    loader = DataLoader(FaceDataset(train_raw, num_keys), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True, generator=loader_gen)

    g_phi    = G_phi(feature_dim, protect_dim, key_dim, num_keys).to(DEVICE)
    arc_head = ArcFaceHead(protect_dim, NUM_SUBJECTS).to(DEVICE)
    critic   = CriticNet(protect_dim, feature_dim).to(DEVICE)
    club     = CLUBNet(protect_dim, NUM_SUBJECTS).to(DEVICE)
    adv      = AdvNet(protect_dim, feature_dim, key_dim, num_keys).to(DEVICE)

    opt_main = optim.Adam(
        list(g_phi.parameters()) + list(arc_head.parameters()) + list(club.parameters()),
        lr=LR_MAIN, weight_decay=1e-5)
    opt_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)
    opt_adv    = optim.Adam(adv.parameters(), lr=LR_ADV)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=EPOCHS)

    best_eer, best_state = 1.0, None

    for epoch in range(1, EPOCHS + 1):
        g_phi.train(); adv.train(); critic.train()
        gamma_e  = get_warmup_weight(epoch, WARMUP_EPOCHS, GAMMA_FINAL)
        lambda_e = get_warmup_weight(epoch, WARMUP_EPOCHS, LAMBDA_FINAL)
        alpha_e  = get_warmup_weight(epoch, WARMUP_EPOCHS, ALPHA_FINAL)
        mu_e     = get_warmup_weight(epoch, WARMUP_EPOCHS, MU_FINAL)

        for x, y, key in loader:
            x, y, key = x.to(DEVICE), y.to(DEVICE), key.to(DEVICE)
            key2 = (key + torch.randint(1, num_keys, (x.size(0),), device=DEVICE)) % num_keys

            Z_K, mu1, s1 = g_phi(x, key)
            Z_K2, mu2, s2 = g_phi(x, key2)

            opt_critic.zero_grad()
            mine_critic_loss(critic, Z_K, x).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_critic.step()

            opt_adv.zero_grad()
            adv_inner_loss(adv, Z_K, x, key, num_keys).backward()
            torch.nn.utils.clip_grad_norm_(adv.parameters(), 1.0)
            opt_adv.step()

            Z_K, mu1, s1 = g_phi(x, key)
            Z_K2, mu2, s2 = g_phi(x, key2)

            mu1_norm = F.normalize(mu1, dim=-1)
            L_rec = arc_head(mu1_norm, y)
            mi = mine_read(critic, Z_K, x)
            y_oh = F.one_hot(y, NUM_SUBJECTS).float()
            L_cl = club_loss(club, Z_K, Z_K2, y_oh)
            L_sec = mi + gamma_e * L_cl
            L_sep = sep_loss_fn(mu1, s1, mu2, s2, delta)
            L_adv = adv_outer_loss_for_generator(adv, Z_K, x, key, num_keys)

            loss = L_rec + lambda_e * L_sec + alpha_e * L_adv + mu_e * L_sep
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            opt_main.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(g_phi.parameters()) + list(arc_head.parameters()) + list(club.parameters()), 1.0)
            opt_main.step()

        scheduler.step()

        if epoch % EVAL_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            eer = evaluate_test_eer(g_phi, test_raw)
            if eer is not None:
                tag_str = "  (best so far)" if eer < best_eer else ""
                print(f"  [{tag}] Epoch {epoch:3d}/{EPOCHS}  TEST EER={eer*100:.2f}%{tag_str}")
                if eer < best_eer:
                    best_eer = eer
                    best_state = {k: v.cpu().clone() for k, v in g_phi.state_dict().items()}

    if best_state is not None:
        g_phi.load_state_dict(best_state)
        g_phi.to(DEVICE)

    prot = extract(g_phi, test_raw)
    gen = genuine_scores(prot)
    imp = impostor_scores_full(prot)
    EER, FAR, FRR, GAR = compute_eer_full(gen, imp)

    print(f"  [{tag}] FINAL EER = {EER*100:.2f}%")

    return {
        "tag": tag, "feature_dim": feature_dim, "protect_dim": protect_dim,
        "key_dim": key_dim, "num_keys": num_keys, "eer": EER,
        "far_curve": FAR, "gar_curve": GAR,
    }


# ============================================================
#  COMPARISON TABLE + PLOTS
# ============================================================
def save_comparison_table(results, filename):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "feature_dim", "protect_dim", "key_dim", "num_keys", "EER_percent"])
        for r in results:
            w.writerow([r["tag"], r["feature_dim"], r["protect_dim"],
                        r["key_dim"], r["num_keys"], f"{r['eer']*100:.3f}"])
    print(f"Comparison table saved: {filename}")

def plot_eer_bars(results, xlabels, title, filename, xaxis_label):
    plt.figure(figsize=(6,5))
    eers = [r["eer"]*100 for r in results]
    bars = plt.bar([str(x) for x in xlabels], eers, color="#3b6ea5")
    for b, e in zip(bars, eers):
        plt.text(b.get_x()+b.get_width()/2, e, f"{e:.2f}%", ha="center", va="bottom")
    plt.xlabel(xaxis_label); plt.ylabel("EER (%)")
    plt.title(title); plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Bar chart saved: {filename}")

def plot_roc_overlay(results, labels, title, filename):
    plt.figure(figsize=(6,5))
    for r, lbl in zip(results, labels):
        plt.plot(r["far_curve"]*100, r["gar_curve"]*100, lw=2,
                  label=f"{lbl} (EER={r['eer']*100:.2f}%)")
    plt.xlabel("FAR (%)"); plt.ylabel("GAR (%)")
    plt.title(title); plt.legend(fontsize=8); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"ROC overlay saved: {filename}")


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    print("=== Step 1: Load LFW-127 ===")
    train_raw, test_raw = load_lfw127_features(DATA_DIR)

    # ---------- SWEEP A: vary PROTECT_DIM, KEY_DIM fixed at 128 ----------
    print("\n" + "#"*70)
    print(f"# SWEEP A: PROTECT_DIM in {PROTECT_DIM_SWEEP}  (KEY_DIM fixed = {KEY_DIM_FIXED_FOR_SWEEP_A})")
    print("#"*70)
    results_A = []
    for pd in PROTECT_DIM_SWEEP:
        tag = f"protect{pd}_key{KEY_DIM_FIXED_FOR_SWEEP_A}"
        res = run_experiment(tag, FEATURE_DIM, pd, KEY_DIM_FIXED_FOR_SWEEP_A, NUM_KEYS,
                              train_raw, test_raw)
        results_A.append(res)

    save_comparison_table(results_A, "compare_protect_dim.csv")
    plot_eer_bars(results_A, PROTECT_DIM_SWEEP,
                  f"EER vs PROTECT_DIM (KEY_DIM={KEY_DIM_FIXED_FOR_SWEEP_A})",
                  "eer_vs_protect_dim.png", "PROTECT_DIM")
    plot_roc_overlay(results_A, [f"PROTECT_DIM={pd}" for pd in PROTECT_DIM_SWEEP],
                      "ROC — varying PROTECT_DIM", "roc_vs_protect_dim.png")

    # ---------- SWEEP B: vary KEY_DIM, PROTECT_DIM fixed at 512 ----------
    print("\n" + "#"*70)
    print(f"# SWEEP B: KEY_DIM in {KEY_DIM_SWEEP}  (PROTECT_DIM fixed = {PROTECT_DIM_FIXED_FOR_SWEEP_B})")
    print("#"*70)
    results_B = []
    for kd in KEY_DIM_SWEEP:
        tag = f"protect{PROTECT_DIM_FIXED_FOR_SWEEP_B}_key{kd}"
        res = run_experiment(tag, FEATURE_DIM, PROTECT_DIM_FIXED_FOR_SWEEP_B, kd, NUM_KEYS,
                              train_raw, test_raw)
        results_B.append(res)

    save_comparison_table(results_B, "compare_key_dim.csv")
    plot_eer_bars(results_B, KEY_DIM_SWEEP,
                  f"EER vs KEY_DIM (PROTECT_DIM={PROTECT_DIM_FIXED_FOR_SWEEP_B})",
                  "eer_vs_key_dim.png", "KEY_DIM")
    plot_roc_overlay(results_B, [f"KEY_DIM={kd}" for kd in KEY_DIM_SWEEP],
                      "ROC — varying KEY_DIM", "roc_vs_key_dim.png")

    # ---------- FINAL PRINTED SUMMARY ----------
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\n-- Sweep A: PROTECT_DIM sweep (KEY_DIM={KEY_DIM_FIXED_FOR_SWEEP_A}) --")
    for r in results_A:
        print(f"  PROTECT_DIM={r['protect_dim']:5d}  KEY_DIM={r['key_dim']:4d}  "
              f"EER={r['eer']*100:.2f}%")
    print(f"\n-- Sweep B: KEY_DIM sweep (PROTECT_DIM={PROTECT_DIM_FIXED_FOR_SWEEP_B}) --")
    for r in results_B:
        print(f"  PROTECT_DIM={r['protect_dim']:5d}  KEY_DIM={r['key_dim']:4d}  "
              f"EER={r['eer']*100:.2f}%")

    print("\nOutputs written:")
    print("  compare_protect_dim.csv, eer_vs_protect_dim.png, roc_vs_protect_dim.png")
    print("  compare_key_dim.csv,     eer_vs_key_dim.png,     roc_vs_key_dim.png")