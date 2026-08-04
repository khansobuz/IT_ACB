"""
IT_ACB_EER.py  — IT-ACB for TJU Palmprint (Correct Train/Test Split)
Section 3.1: KIIS  | Section 3.2: CK-SLG | Section 3.3: KAAMS

Correct protocol (same as TIFS papers):
  Train : all 600 subjects, samples 1-5
  Test  : all 600 subjects, samples 6-10
  EER computed on test samples only — never seen during training

NEW (added after EER, computed ONCE on the trained model / test set,
NOT every epoch — these are final privacy/security evaluation metrics,
exactly the way TIFS-style biometric template protection papers report
them in a single results table):

  Unlinkability:
    - Intra-Key Similarity   (same subject, same key)        -> want HIGH
    - Inter-Key Similarity   (same subject, different key)   -> want LOW
    - Inter-Record Similarity(different subjects)             -> want LOW

  Irreversibility (attack simulation using the trained AdvNet
  fed the protected template + a WRONG key):
    - Reconstruction Error (L2)      -> want HIGH (attack fails)
    - SSIM (reshaped feature "image")-> want LOW  (attack fails)
    - Identity Recovery Rate (%)     -> want LOW  (1-NN identification
                                         of reconstructed x_hat against
                                         gallery of true features)
    - Identity Leakage Rate (%)      -> want LOW  (cosine(x_hat, x_true)
                                         exceeds the EER decision threshold)

Usage: python IT_ACB_EER_full.py
Requires: pip install scikit-image   (for SSIM)
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

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAVE_SSIM = True
except ImportError:
    HAVE_SSIM = False
    print("[WARN] scikit-image not installed -> SSIM will be skipped. "
          "Install with: pip install scikit-image")

# ============================================================
#  SETTINGS
# ============================================================
DATA_DIR      = r"C:\Users\khanm\Desktop\UESTC\PhD. Paper\5th_paper_IEEE_TIFS\PolyU_data"
NUM_SUBJECTS  = 386
NUM_SAMPLES   = 10
TRAIN_SAMPLES = [1, 2, 3, 4, 5, 6, 7, 8]     # samples used for training
TEST_SAMPLES  = [9, 10]                      # samples used for EER evaluation only
FEATURE_DIM   = 512
PROTECT_DIM   = 512
NUM_KEYS      = 10
KEY_DIM       = 64
EPOCHS        = 200
BATCH_SIZE    = 64
LR_MAIN       = 1e-4
LR_CRITIC     = 1e-4
LR_ADV        = 1e-4
DELTA_PER_DIM = 0.5
DELTA         = DELTA_PER_DIM * PROTECT_DIM
GAMMA         = 0.5
LAMBDA        = 0.5
ALPHA_MAX     = 3.0   # was 1.5 -- pushed higher now that the gradient-flow
                       # bug is fixed and EER didn't move at all going
                       # 0.3->1.5, meaning there's headroom before it starts
                       # trading off recognition accuracy
ALPHA_WARMUP_EPOCHS = 20  # ramp in over the first 20 epochs so the stronger
                           # defense doesn't destabilize the generator before
                           # it has learned a good recognition embedding
N_ADV_TRIES   = 5     # was 3 -- more wrong-key attempts per outer step gives
                       # a tighter estimate of the attacker's best-case guess
ADV_INNER_STEPS = 3    # was implicitly 1 -- train the adversary MULTIPLE
                        # steps per batch before the generator has to defend
                        # against it. A weak/undertrained attacker gives a
                        # false sense of security: the generator only needs
                        # to be as good as beating whatever the adversary
                        # currently is. Training the adversary closer to its
                        # own optimum each round makes the generator's
                        # eventual defense meaningful against a strong attack.
MU            = 0.5
ARC_SCALE     = 64.0
ARC_MARGIN    = 0.35
EER_STEP      = 0.001
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reshape factors for treating a 512-d feature vector as a 2D "image" for SSIM
SSIM_H, SSIM_W = 16, 32   # 16 * 32 = 512 = PROTECT_DIM

# keys used to probe unlinkability (any two distinct keys from NUM_KEYS)
KEY_A = 0
KEY_B = 1

# ----- mu-separation (unlinkability) loss -----
# The KL-based sep_loss_fn alone can be "cheated" by inflating sigma instead
# of actually separating mu across keys (mu is all that's used at test time,
# since extract() runs deterministic=True). We add a direct hinge loss on the
# cosine similarity between normalized mu1/mu2 that can't be gamed via sigma.
MU_COS_TARGET   = 0.0    # push cos(mu1, mu2) toward <= this (0 = near-orthogonal)
LAMBDA_MUSEP_MAX = 8.0   # final weight — must be strong enough to compete with
                          # ArcFace (ARC_SCALE=64), which otherwise dominates
                          # and keeps mu nearly key-invariant
MUSEP_WARMUP_EPOCHS = 30 # ramp the weight in AFTER ArcFace has mostly converged,
                          # instead of fighting both objectives from epoch 1

# ----- key-embedding-table separation (fixes uneven per-pair unlinkability) -----
# NOTE: G_phi.key_emb is now an exactly orthogonal, FROZEN table (see G_phi
# __init__), so this loss will naturally sit at ~0 by construction and is
# kept only as a runtime sanity check (KeySep in the epoch log should be
# ~0.00 from epoch 1 onward -- if it isn't, something is wrong with the
# orthogonal init). It's no longer what's doing the separating work.
LAMBDA_KEYSEP = 4.0
KEYSEP_TARGET = 0.0       # push cos(key_i, key_j) toward <= this for all i != j

# ----- explicit full-pairwise mu separation (fixes stubborn key-pair collisions) -----
# Even with a perfectly orthogonal frozen key_emb table, the downstream
# shared MLP can still learn to collapse a handful of specific key
# directions close together -- mu_cos_sep_loss only ever sees a randomly
# sampled pair per training example, and averaging over a batch dilutes the
# gradient for stubborn pairs among many already-fine ones (hard-example
# problem, not a coverage problem). This computes mu for ALL NUM_KEYS keys
# on a small subset of each batch, runs the loss over ALL C(NUM_KEYS,2)
# pairs explicitly (no random sampling, no dilution), and keeps only the
# worst FULL_KEYSEP_TOPK_FRAC fraction of violations so gradient
# concentrates on exactly the pairs that need it.
FULL_KEYSEP_SUBSET       = 8     # samples per batch used for the explicit all-pairs check
FULL_KEYSEP_TOPK_FRAC    = 0.3   # focus on the hardest ~30% of the 45 pairs
LAMBDA_FULL_KEYSEP_MAX   = 6.0
FULL_KEYSEP_WARMUP_EPOCHS = 30   # same warm-in idea as mu_cos_sep_loss

print(f"Device: {DEVICE}  |  ArcFace s={ARC_SCALE} m={ARC_MARGIN}")
print(f"Train samples: {TRAIN_SAMPLES}  |  Test samples: {TEST_SAMPLES}")
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

# ============================================================
#  LOAD DATA — separate train and test
# ============================================================
def load_tju_features(data_dir):
    train_raw = {}
    test_raw  = {}
    for i in range(1, NUM_SUBJECTS + 1):
        for j in range(1, NUM_SAMPLES + 1):
            fp = os.path.join(data_dir, f"{i}_{j}.mat")
            if not os.path.exists(fp):
                continue
            A = sio.loadmat(fp)['feature'].flatten().astype(np.float32)
            if j in TRAIN_SAMPLES:
                train_raw[(i, j)] = A
            elif j in TEST_SAMPLES:
                test_raw[(i, j)]  = A
        if i % 100 == 0:
            print(f"  Loaded {i}/{NUM_SUBJECTS} subjects")
    print(f"Train templates: {len(train_raw)}  |  Test templates: {len(test_raw)}")
    return train_raw, test_raw


# ============================================================
#  DATASET — training only
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


class FiLM(nn.Module):
    """Feature-wise Linear Modulation. Injects the key DIRECTLY at this
    layer via a per-channel scale (gamma) and shift (beta) computed from
    the key embedding, instead of relying on the key surviving several
    Linear/LayerNorm transforms after being concatenated once at the
    input. Diagnosis that motivated this: with a provably orthogonal,
    frozen key embedding table (verified mean/max pairwise cos = 0.000000
    at init), 4 out of 45 key pairs STILL ended up at 0.85-0.95 mu
    similarity after training -- proof the entanglement was happening
    inside the network, not from the key input itself. Concatenate-once
    lets a large 512-dim feature vector dilute a 64-dim key signal through
    depth; FiLM re-applies the key's influence at every layer so it can't
    be "smoothed away."
    Initialized near-identity (gamma~1, beta~0) so early training starts
    close to the old concatenation behavior and the key's steering effect
    grows in as gamma/beta are learned, rather than injecting noise from
    step 1.
    """
    def __init__(self, key_dim, feat_dim):
        super().__init__()
        self.to_gamma = nn.Linear(key_dim, feat_dim)
        self.to_beta  = nn.Linear(key_dim, feat_dim)
        nn.init.zeros_(self.to_gamma.weight); nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight);  nn.init.zeros_(self.to_beta.bias)

    def forward(self, h, k):
        return self.to_gamma(k) * h + self.to_beta(k)


class G_phi(nn.Module):
    """Section 3.1: Key-conditioned stochastic generator."""
    def __init__(self):
        super().__init__()
        self.key_emb = nn.Embedding(NUM_KEYS, KEY_DIM)
        # FIX 1: previously this embedding table was freely learned, and
        # despite a direct key_embedding_sep_loss pushing every pair apart
        # (weighted, LAMBDA_KEYSEP), some specific pairs (e.g. keys 6&7,
        # 1&9, 3&5) still ended up highly correlated (cos up to 0.85) after
        # training -- a hinge loss averaged over 45 pairs can satisfy the
        # mean while leaving individual pairs unseparated (whack-a-mole).
        # Since KEY_DIM=64 >= NUM_KEYS=10, we can instead construct an
        # EXACTLY orthogonal set of key vectors via QR decomposition and
        # FREEZE them. This guarantees cos(key_i, key_j) ~= 0 for every
        # single one of the 45 pairs, permanently, independent of anything
        # that happens during training -- no more whack-a-mole at the INPUT.
        with torch.no_grad():
            q, _ = torch.linalg.qr(torch.randn(KEY_DIM, KEY_DIM))
            self.key_emb.weight.copy_(q[:NUM_KEYS])
        self.key_emb.weight.requires_grad_(False)

        # FIX 2: even with perfectly orthogonal inputs, the network still
        # collapsed 4/45 pairs together downstream -- so the key is now
        # injected via FiLM at EVERY hidden layer (see FiLM docstring above)
        # instead of concatenated once at the input.
        self.fc1 = nn.Linear(FEATURE_DIM, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.mu_head    = nn.Linear(512, PROTECT_DIM)
        self.sigma_head = nn.Sequential(nn.Linear(512, PROTECT_DIM), nn.Softplus())
        self.apply(_init)  # only touches the Linear layers defined above

        # FiLM layers are created AFTER self.apply(_init) so their careful
        # near-identity initialization isn't overwritten by the generic
        # xavier_uniform_ init that _init applies to nn.Linear.
        self.film1 = FiLM(KEY_DIM, 1024)
        self.film2 = FiLM(KEY_DIM, 512)

    def forward(self, x, key, deterministic=False):
        k = self.key_emb(key)
        h = F.relu(self.film1(self.ln1(self.fc1(x)), k))
        h = F.relu(self.film2(self.ln2(self.fc2(h)), k))
        mu    = self.mu_head(h)
        sigma = torch.clamp(self.sigma_head(h), 1e-4, 10.0)
        if deterministic:
            return mu, mu, sigma
        Z_K = mu + sigma * torch.randn_like(mu)
        return Z_K, mu, sigma


class ArcFaceHead(nn.Module):
    """ArcFace angular margin loss — key improvement for EER."""
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
    sigma — it forces the DETERMINISTIC output (mu, what's actually used
    at test time) to differ across keys."""
    m1  = F.normalize(mu1, dim=-1)
    m2  = F.normalize(mu2, dim=-1)
    cos = (m1 * m2).sum(-1)
    return F.relu(cos - target).mean()

def key_embedding_sep_loss(g_phi, target=KEYSEP_TARGET):
    """Direct regularizer on g_phi's key embedding TABLE (NUM_KEYS x KEY_DIM),
    not on a per-batch sample of key pairs. Pushes EVERY pair of the 10 key
    vectors apart, every step -- this is what actually fixes uneven
    per-pair unlinkability (e.g. mu_cos_sep_loss looking fine "on average"
    while one specific pair like key 0 vs key 1 stays entangled)."""
    W  = g_phi.key_emb.weight               # [NUM_KEYS, KEY_DIM]
    Wn = F.normalize(W, dim=-1)
    sim = Wn @ Wn.T                          # [NUM_KEYS, NUM_KEYS] pairwise cosine
    mask = ~torch.eye(NUM_KEYS, dtype=torch.bool, device=W.device)
    off_diag = sim[mask]
    return F.relu(off_diag - target).mean()

def full_pairwise_mu_sep_loss(g_phi, x_subset, target=MU_COS_TARGET,
                               topk_frac=FULL_KEYSEP_TOPK_FRAC):
    """Explicitly computes mu for EVERY one of the NUM_KEYS keys on a small
    subset of samples, then applies a hard-example-mining hinge loss over
    ALL C(NUM_KEYS,2) pairs -- no random sampling, no batch-average dilution.
    Only the worst topk_frac fraction of the 45 pair-violations contribute
    to the loss, so gradient concentrates on exactly the pairs (like the
    (7,8) collision we saw) that mu_cos_sep_loss's random sampling wasn't
    fixing, instead of being averaged away among the ~41 pairs that were
    already fine."""
    B = x_subset.size(0)
    mu_all = []
    for k in range(NUM_KEYS):
        key_batch = torch.full((B,), k, dtype=torch.long, device=x_subset.device)
        _, mu, _ = g_phi(x_subset, key_batch, deterministic=True)
        mu_all.append(F.normalize(mu, dim=-1))
    mu_all = torch.stack(mu_all, dim=0)  # [NUM_KEYS, B, PROTECT_DIM]

    violations = []
    for ka, kb in combinations(range(NUM_KEYS), 2):
        cos = (mu_all[ka] * mu_all[kb]).sum(-1)      # [B]
        violations.append(F.relu(cos - target))
    violations = torch.stack(violations, dim=0).flatten()  # [45 * B]
    k = max(1, int(topk_frac * violations.numel()))
    topk_viol, _ = torch.topk(violations, k)
    return topk_viol.mean()

def adv_inner_loss(adv, z, x, key):
    """Attacker's own training objective. Targets BOTH cosine similarity
    and L2 distance so the trained adversary is a strong, well-rounded
    attacker on every axis it'll later be evaluated on -- a defense that
    only ever saw a cosine-only attacker during training would be blind
    to an L2-style reconstruction attack at test time."""
    B  = z.size(0)
    wk = (key + torch.randint(1, NUM_KEYS, (B,), device=DEVICE)) % NUM_KEYS
    x_hat = adv(z.detach(), wk)
    cos   = F.cosine_similarity(
        F.normalize(x_hat, dim=-1), F.normalize(x, dim=-1), dim=-1)
    l2    = torch.norm(F.normalize(x_hat, dim=-1) - F.normalize(x, dim=-1), dim=-1)
    return (1.0 - cos).mean() + 0.5 * l2.mean()

def adv_outer_loss_for_generator(adv, z, x, key, n_tries=N_ADV_TRIES):
    """Generator's defense objective against the reconstruction attack.

    BUGFIX (was silently a no-op before):
    - The original code called adv(z.detach(), wk) and then also detached
      x_hat before the cosine similarity. That severs the autograd graph
      completely -- L_adv still got computed and logged, but ZERO gradient
      ever reached g_phi from it, no matter how ALPHA was tuned. The
      generator was never actually trained to resist wrong-key
      reconstruction, which is exactly why the attacker (Adv) kept getting
      BETTER over training (0.738 -> 0.798) and Identity Leakage sat at
      100% regardless of any other change.
    - Also switched torch.min -> torch.max across the wrong-key tries:
      the generator must be judged against the attacker's BEST (highest
      cosine similarity) attempt, not its weakest -- min was defending
      against a strawman.

    Now also penalizes L2 similarity in normalized space, not just cosine,
    so the generator is pushed to defend against the same metrics
    (Reconstruction Error, SSIM, Identity Recovery/Leakage) it's graded on
    at evaluation time -- defending against cosine alone left L2-based
    reconstruction quality high even after the cosine-only fix.
    """
    B = z.size(0)
    worst_score = torch.full((B,), -1e9, device=DEVICE)  # attacker's best (max) combined score so far
    for _ in range(n_tries):
        wk    = (key + torch.randint(1, NUM_KEYS, (B,), device=DEVICE)) % NUM_KEYS
        x_hat = adv(z, wk)  # NOT detached -- gradient must flow back into g_phi
        x_hat_n = F.normalize(x_hat, dim=-1)
        x_n     = F.normalize(x,     dim=-1)
        cos   = F.cosine_similarity(x_hat_n, x_n, dim=-1)
        l2    = torch.norm(x_hat_n - x_n, dim=-1)
        # attacker "succeeds" when cos is high AND l2 is low, so combine as
        # a single score the generator must push down for its worst case
        score = cos - 0.5 * l2
        worst_score = torch.max(worst_score, score)
    return worst_score.mean()


# ============================================================
#  TRAINING — on train_raw only
# ============================================================
def train(train_raw):
    print("\n=== Training IT-ACB ===")
    loader = DataLoader(PalmprintDataset(train_raw), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True)

    g_phi    = G_phi().to(DEVICE)
    arc_head = ArcFaceHead().to(DEVICE)
    critic   = CriticNet().to(DEVICE)
    club     = CLUBNet().to(DEVICE)
    adv      = AdvNet().to(DEVICE)

    # Sanity check: confirm the key embedding table is exactly orthogonal
    # (worst pair should be ~0.00, not up to 0.85 like the old learned table)
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
    scheduler  = optim.lr_scheduler.StepLR(opt_main, step_size=80, gamma=0.1)

    for epoch in range(1, EPOCHS + 1):
        g_phi.train(); adv.train(); critic.train()
        t_rec = t_sec = t_adv = t_sep = t_musep = t_keysep = 0.0

        # warm-in: 0 for the first MUSEP_WARMUP_EPOCHS, then linearly ramp
        # up to LAMBDA_MUSEP_MAX over the following MUSEP_WARMUP_EPOCHS
        if epoch <= MUSEP_WARMUP_EPOCHS:
            lambda_musep = 0.0
        else:
            ramp = min(1.0, (epoch - MUSEP_WARMUP_EPOCHS) / MUSEP_WARMUP_EPOCHS)
            lambda_musep = LAMBDA_MUSEP_MAX * ramp

        # warm-in for the adversarial defense weight, same idea: let the
        # generator find a good recognition embedding first, then ramp up
        # the pressure to defend it against reconstruction
        ramp_alpha = min(1.0, epoch / ALPHA_WARMUP_EPOCHS)
        alpha_t    = ALPHA_MAX * ramp_alpha

        # warm-in for the explicit full-pairwise hard-mining separation loss
        if epoch <= FULL_KEYSEP_WARMUP_EPOCHS:
            lambda_full_keysep = 0.0
        else:
            ramp_fks = min(1.0, (epoch - FULL_KEYSEP_WARMUP_EPOCHS) / FULL_KEYSEP_WARMUP_EPOCHS)
            lambda_full_keysep = LAMBDA_FULL_KEYSEP_MAX * ramp_fks

        t_fullkeysep = 0.0

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

            # Adversary inner step(s) -- train it MULTIPLE steps per batch so
            # it's closer to its own optimum before the generator defends
            # against it. A single-step attacker is weak and gives a false
            # sense of security.
            for _ in range(ADV_INNER_STEPS):
                Z_K_adv, _, _ = g_phi(x, key)  # fresh forward each inner step
                opt_adv.zero_grad()
                adv_inner_loss(adv, Z_K_adv, x, key).backward()
                torch.nn.utils.clip_grad_norm_(adv.parameters(), 1.0)
                opt_adv.step()

            # Re-forward
            Z_K,  mu1, s1 = g_phi(x, key)
            Z_K2, mu2, s2 = g_phi(x, key2)

            # ArcFace on L2-normalized mu
            mu1_norm = F.normalize(mu1, dim=-1)
            L_rec    = arc_head(mu1_norm, y)
            mi       = mine_read(critic, Z_K, x)
            y_oh     = F.one_hot(y, NUM_SUBJECTS).float()
            L_cl     = club_loss(club, Z_K, Z_K2, y_oh)
            L_sec    = mi + GAMMA * L_cl
            L_sep    = sep_loss_fn(mu1, s1, mu2, s2)
            L_musep  = mu_cos_sep_loss(mu1, mu2)
            L_keysep = key_embedding_sep_loss(g_phi)
            L_fullkeysep = full_pairwise_mu_sep_loss(g_phi, x[:FULL_KEYSEP_SUBSET])

            # Freeze adv's weights for this call: we want gradient to flow
            # from L_adv back into g_phi (through Z_K), but adv's own params
            # should only ever be updated by its own opt_adv step above.
            for p in adv.parameters():
                p.requires_grad_(False)
            L_adv = adv_outer_loss_for_generator(adv, Z_K, x, key)
            for p in adv.parameters():
                p.requires_grad_(True)

            loss = (L_rec + LAMBDA * L_sec + alpha_t * L_adv + MU * L_sep
                    + lambda_musep * L_musep + LAMBDA_KEYSEP * L_keysep
                    + lambda_full_keysep * L_fullkeysep)

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
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{EPOCHS}] "
                  f"Rec={t_rec/n:.3f}  Sec={t_sec/n:.3f}  "
                  f"Adv={t_adv/n:.3f} (alpha={alpha_t:.2f})  Sep={t_sep/n:.4f}  "
                  f"MuSep={t_musep/n:.4f} (w={lambda_musep:.2f})  "
                  f"KeySep={t_keysep/n:.4f}  "
                  f"FullKeySep={t_fullkeysep/n:.4f} (w={lambda_full_keysep:.2f})  "
                  f"LR={scheduler.get_last_lr()[0]:.1e}")

    print("Training complete.")
    return g_phi, adv


# ============================================================
#  EXTRACT — test samples only, L2-norm mu
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


# ============================================================
#  EER — computed on TEST samples only
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
    print(f"Genuine scores: {len(g)}")
    return np.array(g)

def impostor_scores(prot):
    imp = []
    for idx, (s1,s2) in enumerate(combinations(range(1,NUM_SUBJECTS+1), 2)):
        a1 = [j for j in TEST_SAMPLES if (s1,j) in prot]
        a2 = [j for j in TEST_SAMPLES if (s2,j) in prot]
        for j1 in a1:
            for j2 in a2:
                imp.append(cos_sim(prot[(s1,j1)], prot[(s2,j2)]))
        if idx % 10000 == 0: print(f"  Impostor pairs: {idx}...")
    print(f"Impostor scores: {len(imp)}")
    return np.array(imp)

def compute_eer(gen, imp):
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
    print("      Performance Result (IT-ACB)")
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
    plt.plot(FAR*100, GAR*100, 'b-', lw=2, label='IT-ACB')
    plt.scatter([eer*100],[(1-eer)*100], c='red', zorder=5,
                label=f'EER={eer*100:.2f}%')
    plt.xlabel('FAR (%)'); plt.ylabel('GAR (%)')
    plt.title('ROC — TJU Palmprint (IT-ACB)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('roc_IT_ACB.png', dpi=150)
    print("ROC saved: roc_IT_ACB.png")


def raw_feature_eer_threshold(raw):
    """EER threshold computed on RAW (unprotected) features, not the
    protected/mu-space ones. Needed because the attacker's reconstruction
    x_hat lives in raw-feature space — comparing it against the protected-
    domain EER threshold (~0.42) is comparing two different distributions
    and makes "leakage" look like 100% even when nothing is actually
    leaking (raw palmprint features are naturally highly self-similar
    across the whole dataset, so cosine values sit high regardless)."""
    tmp = {}
    for (s, j), feat in raw.items():
        tmp[(s, j)] = feat / (np.linalg.norm(feat) + 1e-8)
    gen_raw = genuine_scores(tmp)
    imp_raw = impostor_scores(tmp)
    _, _, _, _, raw_thresh = compute_eer(gen_raw, imp_raw)
    return raw_thresh


# ============================================================
#  NEW — SECTION 6: UNLINKABILITY METRICS
#  (Intra-Key / Inter-Key / Inter-Record Similarity)
# ============================================================
def intra_key_similarity(prot):
    """Same subject, SAME key (key=0), different samples. Want HIGH."""
    scores = genuine_scores(prot)   # prot was already extracted with fixed_key=0
    return scores

def inter_key_similarity(g_phi, raw, key_a=KEY_A, key_b=KEY_B):
    """Same subject, SAME sample, but protected with two DIFFERENT keys.
       Want LOW (proves templates from different keys look unrelated).
       NOTE: checks only ONE arbitrary pair -- see inter_key_similarity_all_pairs
       for a much more robust check across every possible key pair, since
       training only pushes pairs apart 'on average' and a single fixed pair
       can look fine or bad somewhat by chance."""
    prot_a = extract(g_phi, raw, fixed_key=key_a)
    prot_b = extract(g_phi, raw, fixed_key=key_b)
    common = set(prot_a.keys()) & set(prot_b.keys())
    scores = [cos_sim(prot_a[k], prot_b[k]) for k in common]
    print(f"Inter-Key pairs (key {key_a} vs {key_b}): {len(scores)}")
    return np.array(scores)

def inter_key_similarity_all_pairs(g_phi, raw):
    """Checks EVERY one of the C(NUM_KEYS,2) key pairs, not just one fixed
    pair. Returns (all_scores_pooled, per_pair_mean_dict, worst_pair).
    This is the metric to trust for a real unlinkability claim -- a single
    fixed pair (like key 0 vs 1) can look good or bad somewhat by chance
    even when the average across all pairs is what training optimizes for.
    """
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
    """Different subjects' protected templates (same key). Want LOW.
       This reuses the impostor score distribution already computed for EER."""
    return imp


# ============================================================
#  NEW — SECTION 7: IRREVERSIBILITY / ATTACK METRICS
#  (Reconstruction Error, SSIM, Identity Recovery Rate, Identity Leakage Rate)
#
#  Attack model: adversary has the protected template Z_K (extracted with the
#  TRUE key) but does NOT know the true key, so it feeds Z_K + a WRONG key
#  into the trained AdvNet and tries to reconstruct the original feature x.
# ============================================================
def _to_ssim_image(vec):
    """Reshape a PROTECT_DIM/FEATURE_DIM vector into a 2D array for SSIM."""
    v = vec.reshape(SSIM_H, SSIM_W)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-8:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)  # normalize to [0,1] for SSIM's data_range=1

def evaluate_privacy_attacks(g_phi, adv, raw, raw_eer_threshold, n_wrong_tries=3):
    """
    Runs the key-mismatch reconstruction attack on every TEST template.
    Returns a dict of aggregated metrics.

    NOTE: raw_eer_threshold must be the EER threshold computed on RAW
    (unprotected) feature cosine similarities — see raw_feature_eer_threshold().
    Do NOT pass the protected-domain EER threshold here; x_hat (the
    reconstruction) and x (the true raw feature) both live in raw-feature
    space, so the comparison threshold must too.
    """
    g_phi.eval(); adv.eval()

    # Build gallery of TRUE (normalized) original features, keyed by subject,
    # used for the 1-NN identity-recovery attack. Uses one sample per subject
    # to keep the search gallery well-defined (first available test sample).
    gallery_ids, gallery_feats = [], []
    seen_subjects = set()
    for (s, j), feat in raw.items():
        if s in seen_subjects:
            continue
        fn = feat / (np.linalg.norm(feat) + 1e-8)
        gallery_ids.append(s)
        gallery_feats.append(fn)
        seen_subjects.add(s)
    gallery_feats = np.stack(gallery_feats, axis=0)  # [num_subjects, FEATURE_DIM]
    gallery_ids   = np.array(gallery_ids)

    recon_errors, ssim_vals, cos_to_true = [], [], []
    correct_identity = 0
    leaked = 0
    total  = 0

    with torch.no_grad():
        for (s, j), feat in raw.items():
            x_np = feat / (np.linalg.norm(feat) + 1e-8)
            x    = torch.tensor(x_np).unsqueeze(0).to(DEVICE)
            true_key = torch.tensor([KEY_A]).to(DEVICE)

            # true protected template
            z, _, _ = g_phi(x, true_key, deterministic=True)

            # attacker tries several wrong keys, keeps its BEST (closest) attempt
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

            # --- Reconstruction Error (L2 distance) ---
            err = float(np.linalg.norm(x_hat_norm - x_np))
            recon_errors.append(err)

            # --- SSIM on reshaped feature "image" ---
            if HAVE_SSIM:
                img_hat  = _to_ssim_image(x_hat_norm)
                img_true = _to_ssim_image(x_np)
                s_val = ssim_fn(img_true, img_hat, data_range=1.0)
                ssim_vals.append(s_val)

            # --- Identity Leakage Rate: does reconstruction still match true
            #     identity above the EER operating threshold? ---
            cos_to_true.append(best_cos)
            if best_cos >= raw_eer_threshold:
                leaked += 1

            # --- Identity Recovery Rate: 1-NN search against gallery ---
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
    print("=== Step 1: Load TJU (train/test split) ===")
    train_raw, test_raw = load_tju_features(DATA_DIR)

    print("\n=== Step 2: Train IT-ACB (on train samples only) ===")
    g_phi, adv = train(train_raw)

    print("\n=== Step 3: Extract TEST templates (unseen during training) ===")
    prot = extract(g_phi, test_raw, fixed_key=KEY_A)

    print("\n=== Step 4: Compute scores ===")
    gen = genuine_scores(prot)
    imp = impostor_scores(prot)

    print("\n=== Step 5: Compute EER ===")
    EER, FAR, FRR, GAR, eer_threshold = compute_eer(gen, imp)
    plot_roc(FAR, GAR, EER)
    print(f"\nFinal EER: {EER*100:.2f}%")

    # --------------------------------------------------------
    # Step 6: Unlinkability metrics (Intra/Inter-Key, Inter-Record)
    # --------------------------------------------------------
    print("\n=== Step 6: Unlinkability Metrics ===")
    intra_key = intra_key_similarity(prot)               # == gen, reused
    inter_key_single = inter_key_similarity(g_phi, test_raw)   # single pair, for reference
    inter_key_all, inter_key_pair_means, worst_pair = inter_key_similarity_all_pairs(g_phi, test_raw)
    inter_rec = inter_record_similarity(imp)              # == imp, reused

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
    # Final combined report — everything AFTER EER, printed once
    # --------------------------------------------------------
    print("\n=========================================================")
    print("           FULL PRIVACY / SECURITY EVALUATION REPORT")
    print("=========================================================")
    print(f"Equal Error Rate (EER)          : {EER*100:.2f} %")
    print(f"EER Threshold (protected domain): {eer_threshold:.4f}")
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