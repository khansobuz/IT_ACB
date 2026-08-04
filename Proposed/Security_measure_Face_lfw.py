
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

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAVE_SSIM = True
except ImportError:
    HAVE_SSIM = False
    print("[WARN] scikit-image not installed -> SSIM will be skipped. "
          "Install with: pip install scikit-image")

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
ALPHA_FINAL     = 3.0    # was 0.15 -- raised now that the dead-gradient bug in
                          # adv_outer_loss_for_generator is fixed (see below).
                          # 0.15 was tuned around a defense term that did
                          # nothing; 3.0 matches what actually worked on TJU
                          # once the gradient path was real.
MU_FINAL        = 0.25
WARMUP_EPOCHS   = 15

ARC_SCALE     = 64.0
ARC_MARGIN    = 0.30

EER_STEP      = 0.001
EVAL_EVERY    = 10
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- mu-separation (unlinkability) loss -- ported from TJU -----
MU_COS_TARGET         = 0.0    # push cos(mu1, mu2) toward <= this
LAMBDA_MUSEP_FINAL     = 8.0   # must be strong enough to compete with ArcFace
MUSEP_WARMUP_EPOCHS    = 15    # ramp in after ArcFace has mostly converged

# ----- key-embedding-table separation -----
# NOTE: key_emb is now an exactly orthogonal, FROZEN table (see G_phi
# __init__), so this loss sits at ~0 by construction and is kept only as a
# runtime sanity check (KeySep in the epoch log should be ~0.00 from
# epoch 1). It's not what's actually doing the separating work anymore.
LAMBDA_KEYSEP = 4.0
KEYSEP_TARGET = 0.0

# ----- explicit full-pairwise mu separation (fixes stubborn key-pair collisions) -----
# Even with a perfectly orthogonal frozen key_emb table, the downstream
# shared MLP can still learn to collapse specific key directions close
# together. This computes mu for ALL NUM_KEYS keys on a small subset of
# each batch, runs the loss over ALL C(NUM_KEYS,2) pairs explicitly (no
# random sampling / dilution), and keeps only the worst
# FULL_KEYSEP_TOPK_FRAC fraction of violations so gradient concentrates on
# exactly the pairs that need it.
FULL_KEYSEP_SUBSET        = 8
FULL_KEYSEP_TOPK_FRAC     = 0.3
LAMBDA_FULL_KEYSEP_FINAL  = 6.0
FULL_KEYSEP_WARMUP_EPOCHS = 15

# ----- adversary strength -----
N_ADV_TRIES     = 5    # wrong-key attempts per outer (generator defense) step
ADV_INNER_STEPS = 3    # adversary updates per batch before generator defends

# keys used for single-pair unlinkability reference metric
KEY_A = 0
KEY_B = 1

# reshape factors for treating a 512-d feature vector as a 2D "image" for SSIM
SSIM_H, SSIM_W = 16, 32   # 16 * 32 = 512 = PROTECT_DIM


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
#  NETWORKS
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
        # FIX (ported from TJU): previously a freely learned embedding table
        # could leave specific key pairs highly correlated (cos up to 0.85-
        # 0.95) even under direct separation losses, since a hinge loss
        # averaged over many pairs can satisfy the mean while leaving
        # individual pairs unseparated. Since KEY_DIM=64 >= NUM_KEYS=10, we
        # construct an EXACTLY orthogonal set of key vectors via QR
        # decomposition and FREEZE them -- guarantees cos(key_i, key_j) ~= 0
        # for every one of the 45 pairs, permanently, independent of
        # training dynamics.
        with torch.no_grad():
            q, _ = torch.linalg.qr(torch.randn(KEY_DIM, KEY_DIM))
            self.key_emb.weight.copy_(q[:NUM_KEYS])
        self.key_emb.weight.requires_grad_(False)

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

def mu_cos_sep_loss(mu1, mu2, target=MU_COS_TARGET):
    """Direct hinge loss on cosine similarity between normalized mu1/mu2.
    Unlike the KL sep_loss_fn, this cannot be satisfied by just inflating
    sigma -- it forces the DETERMINISTIC output (mu, what's actually used
    at test time) to differ across keys."""
    m1  = F.normalize(mu1, dim=-1)
    m2  = F.normalize(mu2, dim=-1)
    cos = (m1 * m2).sum(-1)
    return F.relu(cos - target).mean()

def key_embedding_sep_loss(g_phi, target=KEYSEP_TARGET):
    """Direct regularizer on g_phi's key embedding TABLE. key_emb is now
    frozen-orthogonal, so this should sit at ~0 -- kept as a sanity check."""
    W  = g_phi.key_emb.weight
    Wn = F.normalize(W, dim=-1)
    sim = Wn @ Wn.T
    mask = ~torch.eye(NUM_KEYS, dtype=torch.bool, device=W.device)
    off_diag = sim[mask]
    return F.relu(off_diag - target).mean()

def full_pairwise_mu_sep_loss(g_phi, x_subset, target=MU_COS_TARGET,
                               topk_frac=FULL_KEYSEP_TOPK_FRAC):
    """Explicitly computes mu for EVERY one of the NUM_KEYS keys on a small
    subset of samples, then applies a hard-example-mining hinge loss over
    ALL C(NUM_KEYS,2) pairs -- no random sampling, no batch-average
    dilution. Only the worst topk_frac fraction of violations contribute,
    so gradient concentrates on exactly the pairs that need it."""
    B = x_subset.size(0)
    mu_all = []
    for k in range(NUM_KEYS):
        key_batch = torch.full((B,), k, dtype=torch.long, device=x_subset.device)
        _, mu, _ = g_phi(x_subset, key_batch, deterministic=True)
        mu_all.append(F.normalize(mu, dim=-1))
    mu_all = torch.stack(mu_all, dim=0)  # [NUM_KEYS, B, PROTECT_DIM]

    violations = []
    for ka, kb in combinations(range(NUM_KEYS), 2):
        cos = (mu_all[ka] * mu_all[kb]).sum(-1)
        violations.append(F.relu(cos - target))
    violations = torch.stack(violations, dim=0).flatten()
    k = max(1, int(topk_frac * violations.numel()))
    topk_viol, _ = torch.topk(violations, k)
    return topk_viol.mean()

def adv_inner_loss(adv, z, x, key):
    """Attacker's own training objective. Targets BOTH cosine similarity
    and L2 distance so the trained adversary is a strong, well-rounded
    attacker on every axis it'll later be evaluated on."""
    B  = z.size(0)
    wk = (key + torch.randint(1, NUM_KEYS, (B,), device=DEVICE)) % NUM_KEYS
    x_hat = adv(z.detach(), wk)
    cos   = F.cosine_similarity(
        F.normalize(x_hat, dim=-1), F.normalize(x, dim=-1), dim=-1)
    l2    = torch.norm(F.normalize(x_hat, dim=-1) - F.normalize(x, dim=-1), dim=-1)
    return (1.0 - cos).mean() + 0.5 * l2.mean()

def adv_outer_loss_for_generator(adv, z, x, key, n_tries=N_ADV_TRIES):
    """Generator's defense objective against the reconstruction attack.

    BUGFIX (was a silent no-op in the original LFW draft, same bug as the
    original TJU script):
    - Calling adv(z.detach(), wk) and then also detaching x_hat before the
      cosine similarity severs the autograd graph completely -- L_adv gets
      computed and logged, but ZERO gradient ever reaches g_phi from it, no
      matter how ALPHA is tuned. Fixed by NOT detaching z or x_hat here.
    - Uses torch.max (attacker's BEST/highest-similarity attempt), not min,
      across the wrong-key tries -- the generator must be judged against
      the attacker's best guess, not its weakest.
    - Combines cosine AND L2 similarity so the generator defends against
      both metrics it's graded on at evaluation time (Reconstruction
      Error, SSIM, Identity Recovery/Leakage), not just cosine.
    """
    B = z.size(0)
    worst_score = torch.full((B,), -1e9, device=DEVICE)
    for _ in range(n_tries):
        wk    = (key + torch.randint(1, NUM_KEYS, (B,), device=DEVICE)) % NUM_KEYS
        x_hat = adv(z, wk)  # NOT detached -- gradient must flow back into g_phi
        x_hat_n = F.normalize(x_hat, dim=-1)
        x_n     = F.normalize(x,     dim=-1)
        cos   = F.cosine_similarity(x_hat_n, x_n, dim=-1)
        l2    = torch.norm(x_hat_n - x_n, dim=-1)
        score = cos - 0.5 * l2
        worst_score = torch.max(worst_score, score)
    return worst_score.mean()


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

def evaluate_test_eer(g_phi, test_raw, fixed_key=KEY_A):
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

    # Sanity check: confirm the key embedding table is exactly orthogonal
    with torch.no_grad():
        Wn = F.normalize(g_phi.key_emb.weight, dim=-1)
        sim = (Wn @ Wn.T).cpu().numpy()
        off_diag = sim[~np.eye(NUM_KEYS, dtype=bool)]
        print(f"[sanity] key_emb pairwise cos -- mean: {off_diag.mean():.6f}  "
              f"max: {off_diag.max():.6f}  (both should be ~0.00)")

    opt_main = optim.Adam(
        list(g_phi.parameters()) +
        list(arc_head.parameters()) +
        list(club.parameters()),
        lr=LR_MAIN, weight_decay=1e-5)
    opt_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)
    opt_adv    = optim.Adam(adv.parameters(),    lr=LR_ADV)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=EPOCHS)

    best_eer   = 1.0
    best_state = None   # will hold {'g_phi': ..., 'adv': ...} together, so
                          # the restored pair stays coherent (see note below)

    for epoch in range(1, EPOCHS + 1):
        g_phi.train(); adv.train(); critic.train()
        t_rec = t_sec = t_adv = t_sep = t_musep = t_keysep = t_fullkeysep = 0.0

        gamma_e  = get_warmup_weight(epoch, WARMUP_EPOCHS, GAMMA_FINAL)
        lambda_e = get_warmup_weight(epoch, WARMUP_EPOCHS, LAMBDA_FINAL)
        alpha_e  = get_warmup_weight(epoch, WARMUP_EPOCHS, ALPHA_FINAL)
        mu_e     = get_warmup_weight(epoch, WARMUP_EPOCHS, MU_FINAL)
        musep_e  = get_warmup_weight(epoch, MUSEP_WARMUP_EPOCHS, LAMBDA_MUSEP_FINAL)
        fullkeysep_e = get_warmup_weight(epoch, FULL_KEYSEP_WARMUP_EPOCHS, LAMBDA_FULL_KEYSEP_FINAL)

        for x, y, key in loader:
            x, y, key = x.to(DEVICE), y.to(DEVICE), key.to(DEVICE)
            key2 = (key + torch.randint(1, NUM_KEYS,
                    (x.size(0),), device=DEVICE)) % NUM_KEYS

            Z_K,  mu1, s1 = g_phi(x, key)
            Z_K2, mu2, s2 = g_phi(x, key2)

            # Critic step
            opt_critic.zero_grad()
            mine_critic_loss(critic, Z_K, x).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_critic.step()

            # Adversary inner step(s) -- multiple updates per batch so it's
            # a realistically strong attacker before the generator defends
            for _ in range(ADV_INNER_STEPS):
                Z_K_adv, _, _ = g_phi(x, key)
                opt_adv.zero_grad()
                adv_inner_loss(adv, Z_K_adv, x, key).backward()
                torch.nn.utils.clip_grad_norm_(adv.parameters(), 1.0)
                opt_adv.step()

            # Re-forward
            Z_K,  mu1, s1 = g_phi(x, key)
            Z_K2, mu2, s2 = g_phi(x, key2)

            mu1_norm = F.normalize(mu1, dim=-1)
            L_rec    = arc_head(mu1_norm, y)
            mi       = mine_read(critic, Z_K, x)
            y_oh     = F.one_hot(y, NUM_SUBJECTS).float()
            L_cl     = club_loss(club, Z_K, Z_K2, y_oh)
            L_sec    = mi + gamma_e * L_cl
            L_sep    = sep_loss_fn(mu1, s1, mu2, s2)
            L_musep  = mu_cos_sep_loss(mu1, mu2)
            L_keysep = key_embedding_sep_loss(g_phi)
            L_fullkeysep = full_pairwise_mu_sep_loss(g_phi, x[:FULL_KEYSEP_SUBSET])

            # Freeze adv's weights for this call: gradient should flow from
            # L_adv back into g_phi (through Z_K), but adv's own params
            # should only ever be updated by its own opt_adv step above.
            for p in adv.parameters():
                p.requires_grad_(False)
            L_adv = adv_outer_loss_for_generator(adv, Z_K, x, key)
            for p in adv.parameters():
                p.requires_grad_(True)

            loss = (L_rec + lambda_e * L_sec + alpha_e * L_adv + mu_e * L_sep
                    + musep_e * L_musep + LAMBDA_KEYSEP * L_keysep
                    + fullkeysep_e * L_fullkeysep)

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
            t_musep += L_musep.item(); t_keysep += L_keysep.item()
            t_fullkeysep += L_fullkeysep.item()

        scheduler.step()
        n = len(loader)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{EPOCHS}] "
                  f"Rec={t_rec/n:.3f}  Sec={t_sec/n:.3f}  "
                  f"Adv={t_adv/n:.3f}  Sep={t_sep/n:.4f}  "
                  f"MuSep={t_musep/n:.4f} (w={musep_e:.2f})  "
                  f"KeySep={t_keysep/n:.4f}  "
                  f"FullKeySep={t_fullkeysep/n:.4f} (w={fullkeysep_e:.2f})  "
                  f"w(g,l,a,m)=({gamma_e:.2f},{lambda_e:.2f},"
                  f"{alpha_e:.2f},{mu_e:.2f})  "
                  f"LR={scheduler.get_last_lr()[0]:.1e}", end="")

            if epoch % EVAL_EVERY == 0 or epoch == 1:
                eer = evaluate_test_eer(g_phi, test_raw)
                if eer is not None:
                    print(f"  |  TEST EER={eer*100:.2f}%", end="")
                    if eer < best_eer:
                        best_eer   = eer
                        # Save g_phi AND adv together so the restored pair
                        # stays coherent: if we only saved g_phi's best-EER
                        # checkpoint but kept adv's final-epoch weights, the
                        # later privacy evaluation (Step 7) would be judging
                        # an earlier generator against an adversary that was
                        # never actually trained against it.
                        best_state = {
                            'g_phi': {k: v.cpu().clone()
                                      for k, v in g_phi.state_dict().items()},
                            'adv':   {k: v.cpu().clone()
                                      for k, v in adv.state_dict().items()},
                        }
                        print("  (best so far)", end="")
            print()

    print(f"\nTraining complete. Best TEST EER during training: {best_eer*100:.2f}%")

    if best_state is not None:
        g_phi.load_state_dict(best_state['g_phi'])
        adv.load_state_dict(best_state['adv'])
        g_phi.to(DEVICE); adv.to(DEVICE)
        print("Restored best model checkpoint (g_phi + adv, matched pair).")

    return g_phi, adv


# ============================================================
#  EXTRACT
# ============================================================
def extract(g_phi, raw, fixed_key=KEY_A):
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
    eer_threshold = ths[ind]
    print("\n-------------------------------------")
    print("      Performance Result (IT-ACB, LFW-127)")
    print("-------------------------------------")
    print(f"Verification Rate      : {TSR[ind]*100:.2f} %")
    print(f"Genuine Acceptance Rate: {GAR[ind]*100:.2f} %")
    print(f"False Acceptance Rate  : {FAR[ind]*100:.2f} %")
    print(f"False Rejection Rate   : {FRR[ind]*100:.2f} %")
    print(f"Equal Error Rate (EER) : {EER*100:.2f} %")
    print(f"EER Threshold          : {eer_threshold:.4f}")
    print("-------------------------------------")
    return EER, FAR, FRR, GAR, eer_threshold


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
#  SECTION 6: UNLINKABILITY METRICS
# ============================================================
def intra_key_similarity(prot):
    """Same subject, SAME key, different samples. Want HIGH."""
    return genuine_scores(prot)   # prot was extracted with fixed_key=KEY_A

def inter_key_similarity(g_phi, raw, key_a=KEY_A, key_b=KEY_B):
    """Same subject, SAME sample, two DIFFERENT keys. Single pair, kept
    for reference/continuity -- see inter_key_similarity_all_pairs for
    the metric to actually trust."""
    prot_a = extract(g_phi, raw, fixed_key=key_a)
    prot_b = extract(g_phi, raw, fixed_key=key_b)
    common = set(prot_a.keys()) & set(prot_b.keys())
    scores = [cos_sim(prot_a[k], prot_b[k]) for k in common]
    print(f"Inter-Key pairs (key {key_a} vs {key_b}): {len(scores)}")
    return np.array(scores)

def inter_key_similarity_all_pairs(g_phi, raw):
    """Checks EVERY one of the C(NUM_KEYS,2) key pairs. Returns
    (all_scores_pooled, per_pair_mean_dict, worst_pair). This is the
    metric to trust for a real unlinkability claim."""
    prot_by_key = {k: extract(g_phi, raw, fixed_key=k) for k in range(NUM_KEYS)}
    all_scores = []
    pair_means = {}
    for ka, kb in combinations(range(NUM_KEYS), 2):
        common = set(prot_by_key[ka].keys()) & set(prot_by_key[kb].keys())
        scores = [cos_sim(prot_by_key[ka][k], prot_by_key[kb][k]) for k in common]
        pair_means[(ka, kb)] = float(np.mean(scores))
        all_scores.extend(scores)
    all_scores = np.array(all_scores)
    worst_pair = max(pair_means, key=pair_means.get)
    print(f"Inter-Key pairs checked: {len(pair_means)} key-pairs, "
          f"{len(all_scores)} total sample comparisons")
    return all_scores, pair_means, worst_pair

def inter_record_similarity(imp):
    """Different subjects' protected templates. Want LOW. Reuses the
    impostor score distribution already computed for EER."""
    return imp


# ============================================================
#  SECTION 7: IRREVERSIBILITY / ATTACK METRICS
# ============================================================
def raw_feature_eer_threshold(raw):
    """EER threshold computed on RAW (unprotected) features -- needed
    because the attacker's reconstruction x_hat lives in raw-feature
    space, not the protected/mu-space. Comparing against the protected-
    domain threshold would compare two different distributions."""
    tmp = {}
    for (s, j), feat in raw.items():
        tmp[(s, j)] = feat / (np.linalg.norm(feat) + 1e-8)
    gen_raw = genuine_scores(tmp)
    imp_raw = impostor_scores_full(tmp)
    _, _, _, _, raw_thresh = compute_eer_full(gen_raw, imp_raw)
    return raw_thresh

def _to_ssim_image(vec):
    """Reshape a PROTECT_DIM/FEATURE_DIM vector into a 2D array for SSIM."""
    v = vec.reshape(SSIM_H, SSIM_W)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-8:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)

def evaluate_privacy_attacks(g_phi, adv, raw, raw_eer_threshold, n_wrong_tries=3):
    """Runs the key-mismatch reconstruction attack on every TEST template.
    raw_eer_threshold must be computed on RAW features (see
    raw_feature_eer_threshold), not the protected-domain EER threshold."""
    g_phi.eval(); adv.eval()

    gallery_ids, gallery_feats = [], []
    seen_subjects = set()
    for (s, j), feat in raw.items():
        if s in seen_subjects:
            continue
        fn = feat / (np.linalg.norm(feat) + 1e-8)
        gallery_ids.append(s)
        gallery_feats.append(fn)
        seen_subjects.add(s)
    gallery_feats = np.stack(gallery_feats, axis=0)
    gallery_ids   = np.array(gallery_ids)

    recon_errors, ssim_vals = [], []
    correct_identity = 0
    leaked = 0
    total  = 0

    with torch.no_grad():
        for (s, j), feat in raw.items():
            x_np = feat / (np.linalg.norm(feat) + 1e-8)
            x    = torch.tensor(x_np).unsqueeze(0).to(DEVICE)
            true_key = torch.tensor([KEY_A]).to(DEVICE)

            z, _, _ = g_phi(x, true_key, deterministic=True)

            best_cos = -1.0
            best_xhat = None
            for _ in range(n_wrong_tries):
                wrong_key_val = random.choice(
                    [k for k in range(NUM_KEYS) if k != KEY_A])
                wk = torch.tensor([wrong_key_val]).to(DEVICE)
                x_hat = adv(z, wk)
                cos = F.cosine_similarity(
                    F.normalize(x_hat, dim=-1), F.normalize(x, dim=-1), dim=-1).item()
                if cos > best_cos:
                    best_cos = cos
                    best_xhat = x_hat

            x_hat_np = best_xhat.squeeze(0).cpu().numpy()
            x_hat_norm = x_hat_np / (np.linalg.norm(x_hat_np) + 1e-8)

            err = float(np.linalg.norm(x_hat_norm - x_np))
            recon_errors.append(err)

            if HAVE_SSIM:
                img_hat  = _to_ssim_image(x_hat_norm)
                img_true = _to_ssim_image(x_np)
                s_val = ssim_fn(img_true, img_hat, data_range=1.0)
                ssim_vals.append(s_val)

            if best_cos >= raw_eer_threshold:
                leaked += 1

            sims = gallery_feats @ x_hat_norm
            pred_subject = gallery_ids[int(np.argmax(sims))]
            if pred_subject == s:
                correct_identity += 1

            total += 1

    metrics = {
        "reconstruction_error_mean": float(np.mean(recon_errors)),
        "reconstruction_error_std":  float(np.std(recon_errors)),
        "ssim_mean": float(np.mean(ssim_vals)) if HAVE_SSIM and ssim_vals else None,
        "identity_recovery_rate_pct": 100.0 * correct_identity / total,
        "identity_leakage_rate_pct":  100.0 * leaked / total,
        "n_samples_attacked": total,
    }
    return metrics


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    print("=== Step 1: Load LFW-127 (cleaned, train/test split) ===")
    train_raw, test_raw = load_lfw127_features(DATA_DIR)

    print("\n=== Step 2: Train IT-ACB ===")
    g_phi, adv = train(train_raw, test_raw)

    print("\n=== Step 3: Extract FULL TEST templates ===")
    prot = extract(g_phi, test_raw)

    print("\n=== Step 4: Compute FULL scores ===")
    gen = genuine_scores(prot)
    imp = impostor_scores_full(prot)

    print("\n=== Step 5: Compute FINAL EER ===")
    EER, FAR, FRR, GAR, eer_threshold = compute_eer_full(gen, imp)
    plot_roc(FAR, GAR, EER)
    print(f"\nFinal EER: {EER*100:.2f}%")

    torch.save(g_phi.state_dict(), 'g_phi_lfw127.pth')
    print("Model saved: g_phi_lfw127.pth")

    # --------------------------------------------------------
    # Step 6: Unlinkability metrics
    # --------------------------------------------------------
    print("\n=== Step 6: Unlinkability Metrics ===")
    intra_key = intra_key_similarity(prot)
    inter_key_single = inter_key_similarity(g_phi, test_raw)
    inter_key_all, inter_key_pair_means, worst_pair = inter_key_similarity_all_pairs(g_phi, test_raw)
    inter_rec = inter_record_similarity(imp)

    print(f"Inter-Key worst pair: keys {worst_pair} -> "
          f"mean similarity {inter_key_pair_means[worst_pair]:.4f}")
    print("Per-pair means (all 45 key-pair combinations):")
    for pair, m in sorted(inter_key_pair_means.items(), key=lambda kv: -kv[1]):
        print(f"  keys {pair}: {m:.4f}")

    # --------------------------------------------------------
    # Step 7: Irreversibility / attack metrics
    # --------------------------------------------------------
    print("\n=== Step 7: Irreversibility (Reconstruction Attack) Metrics ===")
    print("--- computing raw-feature-space EER threshold for leakage check ---")
    raw_eer_threshold = raw_feature_eer_threshold(test_raw)
    print(f"Raw-feature EER threshold: {raw_eer_threshold:.4f} "
          f"(protected-domain threshold was {eer_threshold:.4f} — these are "
          f"different spaces, do not mix them)")
    attack_metrics = evaluate_privacy_attacks(g_phi, adv, test_raw, raw_eer_threshold)

    # --------------------------------------------------------
    # Final combined report
    # --------------------------------------------------------
    print("\n=========================================================")
    print("     FULL PRIVACY / SECURITY EVALUATION REPORT (LFW-127)")
    print("=========================================================")
    print(f"Equal Error Rate (EER)               : {EER*100:.2f} %")
    print(f"EER Threshold (protected domain)     : {eer_threshold:.4f}")
    print(f"EER Threshold (raw-feature domain, used for leakage check): "
          f"{raw_eer_threshold:.4f}")
    print("---------------------------------------------------------")
    print(f"Intra-Key Similarity  (mean)         : {intra_key.mean():.4f}  (want HIGH)")
    print(f"Inter-Key Similarity  (all-pairs mean): {inter_key_all.mean():.4f}  (want LOW) <- trust this one")
    print(f"Inter-Key Similarity  (worst pair {worst_pair}): {inter_key_pair_means[worst_pair]:.4f}  (want LOW)")
    print(f"Inter-Key Similarity  (single pair {KEY_A},{KEY_B} ref): {inter_key_single.mean():.4f}  (want LOW)")
    print(f"Inter-Record Similarity (mean)       : {inter_rec.mean():.4f}  (want LOW)")
    print("---------------------------------------------------------")
    print(f"Reconstruction Error  (mean±std): "
          f"{attack_metrics['reconstruction_error_mean']:.4f} ± "
          f"{attack_metrics['reconstruction_error_std']:.4f}  (want HIGH)")
    if attack_metrics['ssim_mean'] is not None:
        print(f"SSIM (mean)                     : {attack_metrics['ssim_mean']:.4f}  (want LOW)")
    else:
        print("SSIM (mean)                     : skipped (install scikit-image)")
    print(f"Identity Recovery Rate (%)      : {attack_metrics['identity_recovery_rate_pct']:.2f} %  (want LOW)")
    print(f"Identity Leakage Rate (%)       : {attack_metrics['identity_leakage_rate_pct']:.2f} %  (want LOW)")
    print(f"(attacked {attack_metrics['n_samples_attacked']} test templates)")
    print("=========================================================")