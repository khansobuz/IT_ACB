 

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict

# ============================================================
#  SETTINGS
# ============================================================
LFW_DIR      = r"C:\Users\khanm\Downloads\LFW\LFW\lfw-deepfunneled\lfw-deepfunneled"
PAIRS_TXT    = r"C:\Users\khanm\Downloads\LFW\LFW\pairs.txt"
OUTPUT_DIR   = r"C:\Users\khanm\Desktop\UESTC\PhD. Paper\5th_paper_IEEE_TIFS\Feauture_extraction\LFW127_outlier_removed"
MIN_IMAGES   = 12
MAX_IMAGES   = 12
EER_STEP     = 0.001

ARCFACE_REF = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


# ============================================================
#  PARSE pairs.txt
# ============================================================
def parse_pairs_referenced_images(pairs_path):
    referenced = defaultdict(set)
    with open(pairs_path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = [p for p in line.strip().split('\t') if p != '']
        if len(parts) == 3:
            name, img1, img2 = parts
            referenced[name].add(int(img1)); referenced[name].add(int(img2))
        elif len(parts) == 4:
            name1, img1, name2, img2 = parts
            referenced[name1].add(int(img1)); referenced[name2].add(int(img2))
    print(f"Names referenced in pairs.txt: {len(referenced)}")
    return referenced


# ============================================================
#  SCAN — get ALL available images per subject (>=12), not capped yet
# ============================================================
def scan_lfw_all_images(lfw_dir, referenced_images):
    print(f"Scanning {lfw_dir} ...")
    all_subjects = sorted(os.listdir(lfw_dir))

    valid = []
    for subj in all_subjects:
        subj_path = os.path.join(lfw_dir, subj)
        if not os.path.isdir(subj_path):
            continue
        imgs = sorted([f for f in os.listdir(subj_path)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        if len(imgs) >= MIN_IMAGES:
            ref_nums = referenced_images.get(subj, set())

            def img_number(fname):
                base = fname.rsplit('.', 1)[0]
                try:
                    return int(base.split('_')[-1])
                except ValueError:
                    return -1

            def priority_key(fname):
                num = img_number(fname)
                return (0 if num in ref_nums else 1, num)

            # Keep up to 20 candidates (pairs.txt-prioritized order)
            imgs_sorted = sorted(imgs, key=priority_key)[:20]
            valid.append((subj, imgs_sorted))

    print(f"Subjects with >= {MIN_IMAGES} images : {len(valid)}")
    return valid


# ============================================================
#  LOAD MODELS
# ============================================================
def load_models():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    print("InsightFace detector + recognizer loaded (buffalo_l).")
    return app


def align_face(img, landmarks_5pt, ref=ARCFACE_REF, output_size=112):
    landmarks_5pt = np.array(landmarks_5pt, dtype=np.float32)
    tform, _ = cv2.estimateAffinePartial2D(landmarks_5pt, ref, method=cv2.LMEDS)
    if tform is None:
        return None
    return cv2.warpAffine(img, tform, (output_size, output_size), borderValue=0.0)


def extract_feature_raw(aligned_bgr, app):
    img_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    img_f   = (img_rgb.astype(np.float32) - 127.5) / 127.5
    img_t   = img_f.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    rec_model = app.models['recognition']
    feat = rec_model.session.run(
        None, {rec_model.session.get_inputs()[0].name: img_t})[0][0]
    return feat.astype(np.float32)


def process_image_flip(img_path, app):
    img = cv2.imread(img_path)
    if img is None:
        return None
    faces = app.get(img)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    aligned = align_face(img, face.kps)
    if aligned is None:
        return None
    feat_orig = extract_feature_raw(aligned, app)
    aligned_flip = cv2.flip(aligned, 1)
    feat_flip = extract_feature_raw(aligned_flip, app)
    feat_avg = (feat_orig + feat_flip) / 2.0
    feat_avg = feat_avg / (np.linalg.norm(feat_avg) + 1e-8)
    return feat_avg.astype(np.float32)


def cos_sim(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0


# ============================================================
#  KEY STEP: extract ALL candidates, then remove outliers per subject
# ============================================================
def extract_and_clean(valid_subjects, lfw_dir, app, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_features = {}
    failed = 0
    dropped_total = 0

    print(f"\nExtracting candidates + removing outliers for "
          f"{len(valid_subjects)} subjects...")

    for subj_idx, (subj_name, img_list) in enumerate(valid_subjects):
        subj_id   = subj_idx + 1
        subj_path = os.path.join(lfw_dir, subj_name)

        # Extract features for ALL candidate images (up to 20)
        candidates = []   # (feat, img_name)
        for img_name in img_list:
            img_path = os.path.join(subj_path, img_name)
            feat = process_image_flip(img_path, app)
            if feat is not None:
                candidates.append((feat, img_name))
            else:
                failed += 1

        if len(candidates) < MIN_IMAGES:
            # Not enough valid detections - skip outlier removal, use what we have
            selected = candidates
        else:
            # KEY STEP: compute consistency score for each candidate
            # (average similarity to all OTHER candidates of same subject)
            feats_only = [c[0] for c in candidates]
            consistency = []
            for idx, feat in enumerate(feats_only):
                others = [feats_only[j] for j in range(len(feats_only)) if j != idx]
                avg_sim = np.mean([cos_sim(feat, o) for o in others])
                consistency.append(avg_sim)

            # Sort by consistency (highest first), drop the outliers
            order = np.argsort(consistency)[::-1]
            selected = [candidates[i] for i in order[:MAX_IMAGES]]
            dropped_total += (len(candidates) - MAX_IMAGES)

        for samp_idx, (feat, img_name) in enumerate(selected[:MAX_IMAGES]):
            samp_id = samp_idx + 1
            all_features[(subj_id, samp_id)] = feat
            np.save(os.path.join(output_dir, f"{subj_id}_{samp_id}.npy"), feat)

        if subj_idx % 20 == 0:
            print(f"  {subj_idx+1}/{len(valid_subjects)} subjects  "
                  f"failed={failed}  dropped_so_far={dropped_total}")

    print(f"\nTotal extracted: {len(all_features)}  failed: {failed}  "
          f"outliers dropped: {dropped_total}")
    return all_features


# ============================================================
#  EER
# ============================================================
def compute_eer(all_features, num_subjects):
    print("\n=== Computing Baseline EER (outlier-removed selection) ===")

    gen = []
    for i in range(1, num_subjects + 1):
        avail = [j for j in range(1, MAX_IMAGES+1) if (i,j) in all_features]
        for j1, j2 in combinations(avail, 2):
            a, b = all_features[(i,j1)], all_features[(i,j2)]
            if np.all(a==0) or np.all(b==0):
                continue
            gen.append(cos_sim(a, b))

    imp = []
    for s1, s2 in combinations(range(1, num_subjects+1), 2):
        if (s1,1) in all_features and (s2,1) in all_features:
            a, b = all_features[(s1,1)], all_features[(s2,1)]
            if not (np.all(a==0) or np.all(b==0)):
                imp.append(cos_sim(a, b))

    gen = np.array(gen)
    imp = np.array(imp)
    print(f"Genuine  scores: {len(gen)}  mean={gen.mean():.4f}")
    print(f"Impostor scores: {len(imp)}  mean={imp.mean():.4f}")

    start = min(gen.min(), imp.min())
    stop  = max(gen.max(), imp.max())
    ths   = np.arange(start, stop + EER_STEP, EER_STEP)
    FAR, FRR, GAR = [], [], []
    for t in ths:
        gar = np.mean(gen >= t); far = np.mean(imp >= t)
        GAR.append(gar); FAR.append(far); FRR.append(1-gar)
    FAR = np.array(FAR); FRR = np.array(FRR); GAR = np.array(GAR)
    ind = np.argmin(np.abs(FAR - FRR))
    EER = (FAR[ind] + FRR[ind]) / 2

    print("\n-------------------------------------")
    print("  LFW-127 Outlier-Removed Selection")
    print("-------------------------------------")
    print(f"Genuine Acceptance Rate: {GAR[ind]*100:.2f} %")
    print(f"False Acceptance Rate  : {FAR[ind]*100:.2f} %")
    print(f"False Rejection Rate   : {FRR[ind]*100:.2f} %")
    print(f"Equal Error Rate (EER) : {EER*100:.2f} %")
    print("-------------------------------------")
    return EER, FAR, FRR


def plot_roc(FAR, FRR, eer):
    GAR = 1 - FRR
    plt.figure(figsize=(6,5))
    plt.plot(FAR*100, GAR*100, 'b-', lw=2, label='LFW-127 Outlier-Removed')
    plt.scatter([eer*100], [(1-eer)*100], c='red', zorder=5,
                label=f'EER={eer*100:.2f}%')
    plt.xlabel('FAR (%)'); plt.ylabel('GAR (%)')
    plt.title('ROC — LFW-127 Outlier-Removed')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('roc_lfw127_outlier_removed.png', dpi=150)
    print("ROC saved: roc_lfw127_outlier_removed.png")


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    print("=== Step 1: Parse pairs.txt ===")
    referenced_images = parse_pairs_referenced_images(PAIRS_TXT)

    print("\n=== Step 2: Scan LFW (get up to 20 candidates per subject) ===")
    valid_subjects = scan_lfw_all_images(LFW_DIR, referenced_images)
    if len(valid_subjects) == 0:
        print("ERROR: No subjects found.")
        exit(1)

    print("\n=== Step 3: Load models ===")
    app = load_models()

    print("\n=== Step 4: Extract + remove outliers (same idea as CASIA cleaning) ===")
    all_features = extract_and_clean(valid_subjects, LFW_DIR, app, OUTPUT_DIR)

    print("\n=== Step 5: Compute EER ===")
    EER, FAR, FRR = compute_eer(all_features, len(valid_subjects))
    plot_roc(FAR, FRR, EER)

    print(f"\n=== DONE ===")
    print(f"LFW-127 outlier-removed EER : {EER*100:.2f}%")
    print(f"Compare to previous best (pairs.txt priority): 1.53%")
    print(f"Subjects                    : {len(valid_subjects)}")