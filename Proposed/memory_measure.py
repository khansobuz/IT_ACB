"""
benchmark_efficiency_lfw.py
=============================
Same three metrics (Enrollment time, Verification time, Storage)
but for your LFW-127 IT-ACB model.

Usage: python benchmark_efficiency_lfw.py
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
#  SETTINGS — LFW-127
# ============================================================
DATA_DIR      = r"C:\Users\khanm\Desktop\UESTC\PhD. Paper\5th_paper_IEEE_TIFS\Feauture_extraction\LFW127_outlier_removed"
MODEL_PATH    = r"g_phi_lfw127.pth"
FEATURE_DIM   = 512
PROTECT_DIM   = 512
NUM_KEYS      = 10
KEY_DIM       = 64
FIXED_KEY     = 0
NUM_SUBJECTS  = 127
NUM_SAMPLES   = 12
NUM_TRIALS    = 1000
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")


# ============================================================
#  G_phi — same architecture as LFW-127 training (no residual)
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


# ============================================================
#  LOAD MODEL AND DATA
# ============================================================
def load_trained_model(model_path):
    g_phi = G_phi().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    g_phi.load_state_dict(state)
    g_phi.eval()
    print(f"Loaded trained model: {model_path}")
    return g_phi


def load_sample_features(data_dir, num_subjects, num_samples, n=NUM_TRIALS):
    """Load sample raw .npy features to benchmark on."""
    features = []
    for i in range(1, num_subjects + 1):
        for j in range(1, num_samples + 1):
            fp = os.path.join(data_dir, f"{i}_{j}.npy")
            if os.path.exists(fp):
                A = np.load(fp).astype(np.float32).flatten()
                features.append(A)
            if len(features) >= n:
                break
        if len(features) >= n:
            break
    print(f"Loaded {len(features)} sample features for benchmarking")
    return features


# ============================================================
#  1. ENROLLMENT TIME
# ============================================================
def benchmark_enrollment_time(g_phi, features, fixed_key=FIXED_KEY):
    print(f"\n=== 1. Enrollment Time (n={len(features)} trials) ===")
    times = []
    with torch.no_grad():
        for feat in features:
            fn  = feat / (np.linalg.norm(feat) + 1e-8)
            x   = torch.tensor(fn).unsqueeze(0).to(DEVICE)
            key = torch.tensor([fixed_key]).to(DEVICE)
            start = time.perf_counter()
            mu, _, _ = g_phi(x, key, deterministic=True)
            mu_norm  = F.normalize(mu, dim=-1)
            _ = mu_norm.cpu().numpy()
            end = time.perf_counter()
            times.append((end - start) * 1000)
    times = np.array(times)
    print(f"Mean   : {times.mean():.4f} ms")
    print(f"Std    : {times.std():.4f} ms")
    print(f"Min    : {times.min():.4f} ms")
    print(f"Max    : {times.max():.4f} ms")
    print(f"Median : {np.median(times):.4f} ms")
    return times


# ============================================================
#  2. VERIFICATION TIME
# ============================================================
def cos_sim_timed(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def benchmark_verification_time(g_phi, features, fixed_key=FIXED_KEY, n=NUM_TRIALS):
    print(f"\n=== 2. Verification Time (n={n} trials) ===")
    protected = []
    with torch.no_grad():
        for feat in features[:min(n+1, len(features))]:
            fn  = feat / (np.linalg.norm(feat) + 1e-8)
            x   = torch.tensor(fn).unsqueeze(0).to(DEVICE)
            key = torch.tensor([fixed_key]).to(DEVICE)
            mu, _, _ = g_phi(x, key, deterministic=True)
            mu_norm  = F.normalize(mu, dim=-1)
            protected.append(mu_norm.squeeze(0).cpu().numpy())

    times = []
    for i in range(len(protected) - 1):
        a, b = protected[i], protected[i+1]
        start = time.perf_counter()
        _ = cos_sim_timed(a, b)
        end = time.perf_counter()
        times.append((end - start) * 1_000_000)

    times = np.array(times)
    print(f"Mean   : {times.mean():.4f} us  ({times.mean()/1000:.6f} ms)")
    print(f"Std    : {times.std():.4f} us")
    print(f"Min    : {times.min():.4f} us")
    print(f"Max    : {times.max():.4f} us")
    print(f"Median : {np.median(times):.4f} us")
    return times


# ============================================================
#  3. STORAGE
# ============================================================
def benchmark_storage(protect_dim=PROTECT_DIM, feature_dim=FEATURE_DIM,
                       num_subjects=NUM_SUBJECTS, num_samples=NUM_SAMPLES):
    print(f"\n=== 3. Storage Requirements ===")
    orig_bytes = feature_dim * 4
    orig_kb    = orig_bytes / 1024
    prot_bytes = protect_dim * 4
    prot_kb    = prot_bytes / 1024

    print(f"Original feature dimension  : {feature_dim}")
    print(f"Protected template dimension: {protect_dim}")
    print(f"Original feature size  : {orig_bytes} bytes  ({orig_kb:.3f} KB)")
    print(f"Protected template size: {prot_bytes} bytes  ({prot_kb:.3f} KB)")
    print(f"Storage overhead        : {prot_bytes - orig_bytes} bytes "
          f"({((prot_bytes/orig_bytes)-1)*100:+.1f}%)")

    total_templates = num_subjects * num_samples
    total_orig_mb = (orig_bytes * total_templates) / (1024*1024)
    total_prot_mb = (prot_bytes * total_templates) / (1024*1024)

    print(f"\nFor full dataset ({num_subjects} subjects x {num_samples} samples "
          f"= {total_templates} templates):")
    print(f"  Original total storage  : {total_orig_mb:.2f} MB")
    print(f"  Protected total storage : {total_prot_mb:.2f} MB")

    return {'orig_bytes': orig_bytes, 'prot_bytes': prot_bytes,
            'total_orig_mb': total_orig_mb, 'total_prot_mb': total_prot_mb}


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    print("=== LFW-127 Efficiency Benchmark ===")
    print("=== Load model and sample data ===")
    g_phi = load_trained_model(MODEL_PATH)
    features = load_sample_features(DATA_DIR, NUM_SUBJECTS, NUM_SAMPLES)

    enrollment_times   = benchmark_enrollment_time(g_phi, features)
    verification_times = benchmark_verification_time(g_phi, features)
    storage_info        = benchmark_storage()

    enroll_str  = f"{enrollment_times.mean():.3f} ms"
    verify_str  = f"{verification_times.mean():.3f} us"
    storage_str = f"{storage_info['prot_bytes']} bytes"

    print(f"\n=== SUMMARY (LFW-127) ===")
    print(f"-------------------------------------------------------")
    print(f"{'Metric':<25}{'Value':<20}")
    print(f"-------------------------------------------------------")
    print(f"{'Enrollment time':<25}{enroll_str:<20}")
    print(f"{'Verification time':<25}{verify_str:<20}")
    print(f"{'Template storage':<25}{storage_str:<20}")
    print(f"-------------------------------------------------------")