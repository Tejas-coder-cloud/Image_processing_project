"""C8 Evaluation on Custom Image List

Reuses the exact original C8 implementation from run_stage2_experiment_c8.py
to evaluate a custom list of images with the frozen C8 configuration.

Usage:
    python run_c8_custom_image_list.py --images 00024.ppm,00082.ppm,... or --manifest manifest.csv

Output directory: experiments_phase_n1/c8_custom_evaluation_v1/
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import joblib

from src.config import PipelineConfig, DatasetConfig
from src.run_phase_l_expc import process_all_images, evaluate_one_to_one
from src.run_phase_m2_experiment import compute_color_mask_solidity
from src.pipeline import run_stage1
from src.stage2_segmentation.thresholding import segment_traffic_sign_colors


def load_gtsdb_annotations(gt_path):
    """Load GTSDB annotations from gt.txt file."""
    annotations = {}
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 6:
                filename = parts[0]
                left, top, right, bottom = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                class_id = int(parts[5])
                if filename not in annotations:
                    annotations[filename] = []
                annotations[filename].append({
                    'filename': filename,
                    'left': left,
                    'top': top,
                    'right': right,
                    'bottom': bottom,
                    'class_id': class_id
                })
    return annotations


def compute_iou_xywh(box1, box2):
    """Compute IoU between two boxes in (x, y, w, h) format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to (x1, y1, x2, y2)
    box1_x1, box1_y1, box1_x2, box1_y2 = x1, y1, x1 + w1, y1 + h1
    box2_x1, box2_y1, box2_x2, box2_y2 = x2, y2, x2 + w2, y2 + h2
    
    # Compute intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # Compute union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


# Configuration
OUTPUT_DIR = Path('results/detection_output/c8_evaluation')
MODEL_DIR = Path('experiments_phase_n1/hog_svm_v1')
SVM_THRESHOLD = 0.2

# C8 LAB threshold values (promoted to production defaults)
LAB_RED_A_STAR_C8 = 130
LAB_BLUE_B_STAR_C8 = 155
LAB_YELLOW_B_STAR_C8 = 140

# C8 AR deviation thresholds
M2D_AR_DEV_THRESHOLD = 0.25
M3D_AR_DEV_THRESHOLD = 0.30


def load_validation_images():
    """Load validation image list from split manifest."""
    manifest_path = Path('data/split_manifest.csv')
    val_images = []
    
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == 'val':
                val_images.append(row['image_name'])
    
    return sorted(val_images)


def load_heldout_images():
    """Load held-out image list (00000.ppm through 00019.ppm)."""
    heldout_images = []
    for i in range(20):
        img_name = f"{i:05d}.ppm"
        heldout_images.append(img_name)
    
    return heldout_images


def load_images_from_manifest(manifest_path: Path):
    """Load image list from a custom manifest CSV."""
    images = []
    
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            images.append(row['image'])
    
    return sorted(images)


def run_m3d_with_config(
    cfg: PipelineConfig,
    sample_images: List[str],
    all_gt: Dict,
    gtsdb_dir: Path,
    m2d_ar_thresh: float = 0.42,
    m3d_ar_thresh: float = 0.48,
) -> Tuple[List[Dict], Dict]:
    """Run M3_D pipeline with given configuration."""
    all_records = process_all_images(cfg, sample_images, all_gt, gtsdb_dir)
    
    # Phase L C2 score filter
    c2_survivors = [r for r in all_records if float(r['score_c2']) >= 0.200]
    
    # Phase M1 vertical corridor
    m1_survivors = [r for r in c2_survivors if 0.25 <= float(r['center_y_ratio']) <= 0.70]
    
    # Phase M2_D color rules
    img_cache = {}
    m2_d_survivors = []
    for r in m1_survivors:
        if r['source'] == 'color':
            im_name = r['image']
            if im_name not in img_cache:
                img_bgr = cv2.imread(str(gtsdb_dir / im_name))
                s1 = run_stage1(img_bgr, cfg.stage1)
                enh = s1['enhanced_image']
                masks = segment_traffic_sign_colors(enh, cfg.stage2)
                comb = np.bitwise_or(masks['mask_red'], masks['mask_blue'])
                comb = np.bitwise_or(comb, masks['mask_yellow'])
                img_cache[im_name] = comb
            mask_comb = img_cache[im_name]
            x = int(r['x'])
            y = int(r['y'])
            w = int(r['w'])
            h = int(r['h'])
            crop_m = mask_comb[y:y+h, x:x+w]
            sol = compute_color_mask_solidity(crop_m)
            sym = float(r['edge_symmetry'])
            ar_dev = float(r['ar_deviation'])
            if sol >= 0.20 and sym >= 0.12 and ar_dev <= m2d_ar_thresh:
                m2_d_survivors.append(r)
        else:
            m2_d_survivors.append(r)
    
    # Phase M3_D edge rules
    m3_d_survivors = []
    for r in m2_d_survivors:
        if r['source'] == 'color':
            m3_d_survivors.append(r)
        else:
            sym = float(r['edge_symmetry'])
            ar_dev = float(r['ar_deviation'])
            area = float(r['box_area'])
            if sym >= 0.18 and ar_dev <= m3d_ar_thresh and area <= 15000.0:
                m3_d_survivors.append(r)
    
    return m3_d_survivors, all_records


def evaluate_results(
    m3d_survivors: List[Dict],
    sample_images: List[str],
    all_gt: Dict,
) -> Dict[str, Any]:
    """Evaluate results and return metrics."""
    by_img = {}
    for r in m3d_survivors:
        by_img.setdefault(r['image'], []).append(r)
    
    c_tp = c_fp = c_fn = 0
    per_image_summary = []
    matches_keys = set()
    
    for img_name in sample_images:
        recs = by_img.get(img_name, [])
        det_boxes = [(int(r['x']), int(r['y']), int(r['w']), int(r['h'])) for r in recs]
        gt_boxes = all_gt.get(img_name, [])
        ev = evaluate_one_to_one(det_boxes, gt_boxes, 0.50)
        c_tp += ev['tp']
        c_fp += ev['fp']
        c_fn += ev['fn']
        
        for m in ev['matches']:
            key = f"{img_name}:GT#{m['gt_idx']}(cls{m['class_id']})"
            matches_keys.add(key)
        
        per_image_summary.append({
            'image': img_name,
            'candidates': len(recs),
            'tp': ev['tp'],
            'fp': ev['fp'],
            'fn': ev['fn'],
            'precision': ev['precision'],
            'recall': ev['recall'],
            'f1': ev['f1'],
        })
    
    prec = c_tp / max(1, c_tp + c_fp)
    rec = c_tp / max(1, c_tp + c_fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    
    return {
        'total_candidates': len(m3d_survivors),
        'tp': c_tp,
        'fp': c_fp,
        'fn': c_fn,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'per_image': per_image_summary,
        'matches_keys': matches_keys,
    }


def extract_hog_features_from_crop(
    crop_bgr: np.ndarray,
    hog_config: dict,
) -> np.ndarray:
    """Extract HOG features from a crop (BGR image)."""
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    crop_resized = cv2.resize(crop_gray, (64, 64), interpolation=cv2.INTER_LINEAR)
    
    hog = cv2.HOGDescriptor(
        _winSize=tuple(hog_config['window_size']),
        _blockSize=(hog_config['pixels_per_cell'][0] * hog_config['cells_per_block'][0],
                   hog_config['pixels_per_cell'][1] * hog_config['cells_per_block'][1]),
        _blockStride=(hog_config['pixels_per_cell'][0], hog_config['pixels_per_cell'][1]),
        _cellSize=tuple(hog_config['pixels_per_cell']),
        _nbins=hog_config['orientations'],
    )
    
    features = hog.compute(crop_resized).flatten()
    return features


def svm_filter_candidates(
    candidates: List[Dict],
    img_bgr: np.ndarray,
    svm,
    scaler,
    hog_config: dict,
    threshold: float = 0.0,
) -> List[Dict]:
    """Apply SVM filter to candidate list."""
    filtered = []
    
    for cand in candidates:
        x = int(cand['x'])
        y = int(cand['y'])
        w = int(cand['w'])
        h = int(cand['h'])
        crop = img_bgr[y:y+h, x:x+w]
        
        if crop.size == 0:
            continue
        
        try:
            features = extract_hog_features_from_crop(crop, hog_config)
        except Exception as e:
            continue
        
        features_scaled = scaler.transform(features.reshape(1, -1))
        decision_score = svm.decision_function(features_scaled)[0]
        
        if decision_score >= threshold:
            cand_copy = cand.copy()
            cand_copy['svm_score'] = decision_score
            filtered.append(cand_copy)
    
    return filtered


def main():
    parser = argparse.ArgumentParser(description='C8 Evaluation on Custom Image List')
    parser.add_argument('--images', type=str, help='Comma-separated list of image filenames')
    parser.add_argument('--manifest', type=str, help='Path to CSV manifest with image column')
    parser.add_argument('--validation', action='store_true', help='Use 98 validation images')
    parser.add_argument('--heldout', action='store_true', help='Use 20 held-out images')
    parser.add_argument('--combined', action='store_true', help='Use combined 118 images (98 validation + 20 held-out)')
    
    args = parser.parse_args()
    
    # Determine image list
    if args.validation:
        sample_images = load_validation_images()
        print(f"Using validation images: {len(sample_images)}")
    elif args.heldout:
        sample_images = load_heldout_images()
        print(f"Using held-out images: {len(sample_images)}")
    elif args.combined:
        val_images = load_validation_images()
        heldout_images = load_heldout_images()
        sample_images = sorted(set(val_images + heldout_images))
        print(f"Using combined images: {len(sample_images)} ({len(val_images)} validation + {len(heldout_images)} held-out)")
    elif args.manifest:
        sample_images = load_images_from_manifest(Path(args.manifest))
        print(f"Using images from manifest: {len(sample_images)}")
    elif args.images:
        sample_images = [img.strip() for img in args.images.split(',')]
        print(f"Using provided images: {len(sample_images)}")
    else:
        print("ERROR: Must specify --images, --manifest, --validation, --heldout, or --combined")
        return
    
    # Load GT annotations
    root = Path('.').resolve()
    ds_cfg = DatasetConfig()
    gtsdb_dir = root / ds_cfg.gtsdb_dir
    gt_file = root / ds_cfg.gtsdb_gt_file
    all_gt, _ = load_gtsdb_annotations(gt_file, gtsdb_dir)
    
    gt_count = sum(len(all_gt.get(img, [])) for img in sample_images)
    print(f"Total GT boxes: {gt_count}")
    
    # C8 configuration (exact copy from run_stage2_experiment_c8.py)
    cfg = PipelineConfig()
    cfg.stage2.independent_color_morphology = True
    cfg.stage2.morph_kernel_size = 3
    cfg.stage2.morph_close_iterations = 3
    cfg.stage2.morph_open_iterations = 1
    cfg.stage2.filter9_mode = 'alternative'
    cfg.stage2.filter11_max_area = 8000
    cfg.stage2.nms_iou_threshold = 0.40
    cfg.stage2.hsv_blue_lower = (95, 70, 40)
    cfg.stage2.hsv_blue_upper = (130, 255, 255)
    cfg.stage2.lab_blue_lower = (0, 0, 0)
    cfg.stage2.lab_blue_upper = (255, 255, 125)
    cfg.stage2.hsv_red_lower1 = (0, 70, 40)
    cfg.stage2.hsv_red_lower2 = (155, 70, 40)
    cfg.stage2.hsv_yellow_lower = (14, 70, 50)
    
    # C8 LAB threshold changes
    cfg.stage2.lab_red_lower = (20, LAB_RED_A_STAR_C8, 100)
    cfg.stage2.lab_red_upper = (255, 255, 255)
    cfg.stage2.lab_blue_lower = (20, 60, 20)
    cfg.stage2.lab_blue_upper = (230, 145, LAB_BLUE_B_STAR_C8)
    cfg.stage2.lab_yellow_lower = (40, 100, LAB_YELLOW_B_STAR_C8)
    cfg.stage2.lab_yellow_upper = (255, 160, 255)
    
    print("\nConfiguration (C8 frozen baseline):")
    print(f"  LAB red a*: {LAB_RED_A_STAR_C8}")
    print(f"  LAB blue b*: {LAB_BLUE_B_STAR_C8}")
    print(f"  LAB yellow b*: {LAB_YELLOW_B_STAR_C8}")
    print(f"  HSV saturation: 70")
    print(f"  M2D AR deviation: {M2D_AR_DEV_THRESHOLD}")
    print(f"  M3D AR deviation: {M3D_AR_DEV_THRESHOLD}")
    print(f"  SVM threshold: {SVM_THRESHOLD}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run M3_D pipeline
    print("\nRunning M3_D pipeline...")
    m3d_survivors, all_records = run_m3d_with_config(
        cfg, sample_images, all_gt, gtsdb_dir,
        m2d_ar_thresh=M2D_AR_DEV_THRESHOLD, m3d_ar_thresh=M3D_AR_DEV_THRESHOLD
    )
    
    # Pre-SVM metrics
    pre_svm_metrics = evaluate_results(m3d_survivors, sample_images, all_gt)
    
    print(f"\nPre-SVM metrics:")
    print(f"  Candidates: {pre_svm_metrics['total_candidates']}")
    print(f"  TP: {pre_svm_metrics['tp']}, FP: {pre_svm_metrics['fp']}, FN: {pre_svm_metrics['fn']}")
    
    # Load SVM
    try:
        svm = joblib.load(MODEL_DIR / 'svm_model.joblib')
        scaler = joblib.load(MODEL_DIR / 'scaler.joblib')
        
        with open(MODEL_DIR / 'hog_config.json', 'r') as f:
            hog_config = json.load(f)
        
        print("\nSVM model loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load SVM model: {e}")
        return
    
    # Apply SVM
    svm_filtered = []
    for img_name in sample_images:
        img_bgr = cv2.imread(str(gtsdb_dir / img_name))
        if img_bgr is None:
            continue
        img_candidates = [c for c in m3d_survivors if c['image'] == img_name]
        filtered = svm_filter_candidates(img_candidates, img_bgr, svm, scaler, hog_config, SVM_THRESHOLD)
        svm_filtered.extend(filtered)
    
    # Post-SVM metrics
    post_svm_metrics = evaluate_results(svm_filtered, sample_images, all_gt)
    
    print(f"\nPost-SVM metrics:")
    print(f"  Candidates: {post_svm_metrics['total_candidates']}")
    print(f"  TP: {post_svm_metrics['tp']}, FP: {post_svm_metrics['fp']}, FN: {post_svm_metrics['fn']}")
    print(f"  Precision: {post_svm_metrics['precision']:.4f}")
    print(f"  Recall: {post_svm_metrics['recall']:.4f}")
    print(f"  F1: {post_svm_metrics['f1']:.4f}")
    
    # Save results
    print(f"\nSaving results to: {OUTPUT_DIR}")
    
    metrics = {
        'images_evaluated': len(sample_images),
        'gt_boxes': gt_count,
        'pre_svm_candidates': pre_svm_metrics['total_candidates'],
        'pre_svm_tp': pre_svm_metrics['tp'],
        'pre_svm_fp': pre_svm_metrics['fp'],
        'pre_svm_fn': pre_svm_metrics['fn'],
        'post_svm_candidates': post_svm_metrics['total_candidates'],
        'post_svm_tp': post_svm_metrics['tp'],
        'post_svm_fp': post_svm_metrics['fp'],
        'post_svm_fn': post_svm_metrics['fn'],
        'precision': post_svm_metrics['precision'],
        'recall': post_svm_metrics['recall'],
        'f1': post_svm_metrics['f1'],
        'configuration': {
            'lab_red_a': LAB_RED_A_STAR_C8,
            'lab_blue_b': LAB_BLUE_B_STAR_C8,
            'lab_yellow_b': LAB_YELLOW_B_STAR_C8,
            'hsv_saturation': 70,
            'm2d_ar_dev': M2D_AR_DEV_THRESHOLD,
            'm3d_ar_dev': M3D_AR_DEV_THRESHOLD,
            'svm_threshold': SVM_THRESHOLD,
        },
    }
    
    metrics_file = OUTPUT_DIR / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    per_image_file = OUTPUT_DIR / 'per_image_metrics.csv'
    with open(per_image_file, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image', 'candidates', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])
        w.writeheader()
        w.writerows(post_svm_metrics['per_image'])
    
    # Save image manifest
    manifest_file = OUTPUT_DIR / 'image_manifest.csv'
    with open(manifest_file, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image', 'gt_count'])
        w.writeheader()
        for img in sample_images:
            w.writerow({'image': img, 'gt_count': len(all_gt.get(img, []))})
    
    print(f"\nEvaluation complete!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
