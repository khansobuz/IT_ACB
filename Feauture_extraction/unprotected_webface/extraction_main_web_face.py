 

import os
import shutil
import numpy as np
from itertools import combinations

# ============================================================
#  SETTINGS
# ============================================================
SOURCE_DIR   = r"C:\Users\khanm\Desktop\UESTC\PhD. Paper\5th_paper_IEEE_TIFS\Feauture_extraction\WebFace_features_itacb"
OUTPUT_DIR   = r"C:\Users\khanm\Desktop\UESTC\PhD. Paper\5th_paper_IEEE_TIFS\Feauture_extraction\WebFace_clean_3000"
NUM_SUBJECTS = 10542   # total subjects in source
MAX_IMAGES   = 10
TOP_N        = 3000    # how many clean subjects to keep
EER_STEP     = 0.001


def cos_sim(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0


def load_subject_features(subj_id, source_dir, max_images=MAX_IMAGES):
    feats = {}
    for j in range(1, max_images + 1):
        fp = os.path.join(source_dir, f"{subj_id}_{j}.npy")
        if os.path.exists(fp):
            feats[j] = np.load(fp)
    return feats


def compute_subject_quality(subj_id, source_dir):
    """Average intra-subject similarity — higher = cleaner labels."""
    feats = load_subject_features(subj_id, source_dir)
    feats = {j: f for j, f in feats.items() if not np.all(f == 0)}
    if len(feats) < 2:
        return None, feats
    sims = []
    for (j1, a), (j2, b) in combinations(feats.items(), 2):
        sims.append(cos_sim(a, b))
    if len(sims) == 0:
        return None, feats
    return np.mean(sims), feats


def compute_eer(gen, imp):
    gen = np.array(gen); imp = np.array(imp)
    ths = np.arange(min(gen.min(), imp.min()),
                    max(gen.max(), imp.max()) + EER_STEP, EER_STEP)
    FAR, FRR = [], []
    for t in ths:
        FAR.append(np.mean(imp >= t))
        FRR.append(1 - np.mean(gen >= t))
    FAR = np.array(FAR); FRR = np.array(FRR)
    ind = np.argmin(np.abs(FAR - FRR))
    return (FAR[ind] + FRR[ind]) / 2


# ============================================================
#  STEP 1: SCORE ALL SUBJECTS
# ============================================================
def score_all_subjects(source_dir, num_subjects):
    print("=== Step 1: Scoring subject quality ===")
    qualities = []   # (subj_id, quality, feats_dict)

    for subj_id in range(1, num_subjects + 1):
        q, feats = compute_subject_quality(subj_id, source_dir)
        if q is not None and len(feats) >= 2:
            qualities.append((subj_id, q, feats))
        if subj_id % 2000 == 0:
            print(f"  Scored {subj_id}/{num_subjects} subjects...")

    print(f"Total valid subjects: {len(qualities)}")
    qualities.sort(key=lambda x: x[1], reverse=True)
    return qualities


# ============================================================
#  STEP 2: SAVE TOP-N CLEANEST SUBJECTS
# ============================================================
def save_clean_subset(qualities, top_n, output_dir):
    print(f"\n=== Step 2: Saving top-{top_n} cleanest subjects ===")
    os.makedirs(output_dir, exist_ok=True)

    selected = qualities[:top_n]
    saved_files = 0

    for new_id, (old_subj_id, quality, feats) in enumerate(selected, start=1):
        for samp_idx, (old_samp_id, feat) in enumerate(
                sorted(feats.items()), start=1):
            out_path = os.path.join(output_dir, f"{new_id}_{samp_idx}.npy")
            np.save(out_path, feat)
            saved_files += 1

        if new_id % 500 == 0:
            print(f"  Saved {new_id}/{top_n} subjects...")

    print(f"\nTotal .npy files saved: {saved_files}")
    print(f"Subjects saved        : {len(selected)}")
    print(f"Output folder         : {output_dir}")
    return len(selected)


# ============================================================
#  STEP 3: VERIFY FINAL EER ON SAVED CLEAN SUBSET
# ============================================================
def verify_clean_eer(output_dir, num_subjects, max_images=MAX_IMAGES):
    print(f"\n=== Step 3: Verify EER on saved clean subset ===")

    all_features = {}
    for subj_id in range(1, num_subjects + 1):
        for samp_id in range(1, max_images + 1):
            fp = os.path.join(output_dir, f"{subj_id}_{samp_id}.npy")
            if os.path.exists(fp):
                all_features[(subj_id, samp_id)] = np.load(fp)

    gen, imp = [], []
    for i in range(1, num_subjects + 1):
        avail = [j for j in range(1, max_images+1) if (i,j) in all_features]
        for j1, j2 in combinations(avail, 2):
            gen.append(cos_sim(all_features[(i,j1)], all_features[(i,j2)]))

    # Sample impostor pairs (first 500 subjects to keep it fast)
    subj_list = list(range(1, min(num_subjects+1, 501)))
    for s1, s2 in combinations(subj_list, 2):
        for j in range(1, max_images+1):
            if (s1,j) in all_features and (s2,j) in all_features:
                imp.append(cos_sim(all_features[(s1,j)], all_features[(s2,j)]))

    gen = np.array(gen); imp = np.array(imp)
    print(f"Genuine  : {len(gen)}  mean={gen.mean():.4f}")
    print(f"Impostor : {len(imp)}  mean={imp.mean():.4f}")

    eer = compute_eer(gen, imp)
    print(f"\n-------------------------------------")
    print(f"  Final Clean CASIA-WebFace (Top-3000)")
    print(f"-------------------------------------")
    print(f"Equal Error Rate (EER) : {eer*100:.2f} %")
    print(f"-------------------------------------")
    return eer


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    # Step 1: score all subjects by cleanliness
    qualities = score_all_subjects(SOURCE_DIR, NUM_SUBJECTS)

    # Step 2: save top 3000 cleanest, renumbered 1-3000
    num_saved = save_clean_subset(qualities, TOP_N, OUTPUT_DIR)

    # Step 3: verify final EER
    eer = verify_clean_eer(OUTPUT_DIR, num_saved)

    print(f"\n=== DONE ===")
    print(f"Clean subjects saved : {num_saved}")
    print(f"Final EER            : {eer*100:.2f}%")
    print(f"Output folder        : {OUTPUT_DIR}")
    print(f"\nFor IT-ACB update settings:")
    print(f'  DATA_DIR     = r"{OUTPUT_DIR}"')
    print(f'  NUM_SUBJECTS = {num_saved}')
    print(f'  NUM_SAMPLES  = {MAX_IMAGES}')