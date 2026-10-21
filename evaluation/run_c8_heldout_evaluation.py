"""C8 Held-Out Evaluation

Evaluate the promoted C8 baseline on the held-out images (00000.ppm through 00019.ppm)
to validate generalization to unseen data.

Output directory: experiments_phase_n1/c8_heldout_evaluation_v1/
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import joblib

from src.config import PipelineConfig, DatasetConfig
from evaluate_and_compare import load_gtsdb_annotations, compute_iou_xywh
from src.pipeline import run_stage1
from src.stage2_segmentation.thresholding import segment_traffic_sign_colors
from run_phase_l_expc import process_all_images, evaluate_one_to_one
from run_phase_m2_experiment import compute_color_mask_solidity


# Configuration
OUTPUT_DIR = Path('experiments_phase_n1/c8_heldout_evaluation_v1')
MODEL_DIR = Path('experiments_phase_n1/hog_svm_v1')
SVM_THRESHOLD = 0.2

# C8 AR deviation thresholds
M2D_AR_DEV_THRESHOLD = 0.25
M3D_AR_DEV_THRESHOLD = 0.30

# Held-out image range
HELDOUT_START = 0
HELDOUT_END = 19
HELDOUT_COUNT = 20


def load_heldout_images():
    """Load held-out image list (00000.ppm through 00019.ppm)."""
    heldout_images = []
    for i in range(HELDOUT_START, HELDOUT_END + 1):
        img_name = f"{i:05d}.ppm"
        heldout_images.append(img_name)
    
    return heldout_images


def run_m3d_with_config(
    cfg: PipelineConfig,
    sample_images: List[str],
    all_gt: Dict,
    gtsdb_dir: Path,
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
            if sol >= 0.20 and sym >= 0.12 and ar_dev <= M2D_AR_DEV_THRESHOLD:
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
            if sym >= 0.18 and ar_dev <= M3D_AR_DEV_THRESHOLD and area <= 15000.0:
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
    matches_details = []
    
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
            matches_details.append({
                'image': img_name,
                'gt_idx': m['gt_idx'],
                'class_id': m['class_id'],
                'iou': m['iou'],
            })
        
        per_image_summary.append({
            'image': img_name,
            'candidates': len(recs),
            'tp': ev['tp'],
            'fp': ev['fp'],
            'fn': ev['fn'],
            'precision': ev['precision'],
            'recall': ev['recall'],
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
        'matches_details': matches_details,
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


def create_visual_overlay(
    image_bgr: np.ndarray,
    gt_boxes: List,
    det_boxes: List[Tuple[int, int, int, int]],
    matches: List[Dict[str, Any]],
    output_path: Path,
):
    """Create visual overlay with GT boxes and detections."""
    overlay = image_bgr.copy()
    
    # Draw GT boxes in green
    for gt_box in gt_boxes:
        # Handle both tuple and dict formats
        if isinstance(gt_box, (list, tuple)):
            x, y, w, h = gt_box[:4]
        else:
            x, y, w, h = gt_box['x'], gt_box['y'], gt_box['w'], gt_box['h']
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw detection boxes
    for det_box in det_boxes:
        x, y, w, h = det_box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 1)
    
    # Draw matches in yellow
    for m in matches:
        # Skip drawing matches since we don't have det_x/det_y in matches_details
        pass
    
    cv2.imwrite(str(output_path), overlay)


def main():
    root = Path('.').resolve()
    ds_cfg = DatasetConfig()
    gtsdb_dir = root / ds_cfg.gtsdb_dir
    gt_file = root / ds_cfg.gtsdb_gt_file
    
    if OUTPUT_DIR.exists():
        print(f"ERROR: Output directory already exists: {OUTPUT_DIR}")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_dir = OUTPUT_DIR / 'visual_overlays'
    overlay_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("C8 Held-Out Evaluation")
    print("=" * 80)
    
    heldout_images = load_heldout_images()
    print(f"\nHeld-out images: {len(heldout_images)}")
    print(f"Range: {heldout_images[0]} to {heldout_images[-1]}")
    
    all_gt, _ = load_gtsdb_annotations(gt_file, gtsdb_dir)
    heldout_gt_count = sum(len(all_gt.get(img, [])) for img in heldout_images)
    print(f"Total GT boxes in held-out: {heldout_gt_count}")
    
    # Use promoted C8 configuration from production defaults
    cfg = PipelineConfig()
    cfg.stage2.independent_color_morphology = True
    cfg.stage2.morph_kernel_size = 3
    cfg.stage2.morph_close_iterations = 3
    cfg.stage2.morph_open_iterations = 1
    cfg.stage2.filter9_mode = 'alternative'
    cfg.stage2.filter11_max_area = 8000
    cfg.stage2.nms_iou_threshold = 0.40
    
    # Override HSV saturation to S=70 (C8 configuration)
    cfg.stage2.hsv_blue_lower = (95, 70, 40)
    cfg.stage2.hsv_blue_upper = (130, 255, 255)
    cfg.stage2.hsv_red_lower1 = (0, 70, 40)
    cfg.stage2.hsv_red_lower2 = (155, 70, 40)
    cfg.stage2.hsv_yellow_lower = (14, 70, 50)
    
    print("\n" + "=" * 80)
    print("Configuration Verification")
    print("=" * 80)
    
    print(f"\nLAB red a* threshold: {cfg.stage2.lab_red_lower[1]} (expected: 130)")
    print(f"LAB blue b* threshold: {cfg.stage2.lab_blue_upper[2]} (expected: 155)")
    print(f"LAB yellow b* threshold: {cfg.stage2.lab_yellow_lower[2]} (expected: 140)")
    print(f"HSV saturation: {cfg.stage2.hsv_red_lower1[1]} (expected: 70)")
    
    print("\n" + "=" * 80)
    print("Running Evaluation")
    print("=" * 80)
    
    # Run M3_D
    m3d_survivors, _ = run_m3d_with_config(cfg, heldout_images, all_gt, gtsdb_dir)
    pre_svm_metrics = evaluate_results(m3d_survivors, heldout_images, all_gt)
    
    print(f"\nPre-SVM metrics:")
    print(f"  Candidates: {pre_svm_metrics['total_candidates']}")
    print(f"  TP: {pre_svm_metrics['tp']}, FP: {pre_svm_metrics['fp']}, FN: {pre_svm_metrics['fn']}")
    
    # Load SVM
    try:
        svm = joblib.load(MODEL_DIR / 'svm_model.joblib')
        scaler = joblib.load(MODEL_DIR / 'scaler.joblib')
        
        with open(MODEL_DIR / 'hog_config.json', 'r') as f:
            hog_config = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load SVM model: {e}")
        return
    
    # Apply SVM
    svm_filtered = []
    for img_name in heldout_images:
        img_bgr = cv2.imread(str(gtsdb_dir / img_name))
        if img_bgr is None:
            continue
        img_candidates = [c for c in m3d_survivors if c['image'] == img_name]
        filtered = svm_filter_candidates(img_candidates, img_bgr, svm, scaler, hog_config, SVM_THRESHOLD)
        svm_filtered.extend(filtered)
    
    post_svm_metrics = evaluate_results(svm_filtered, heldout_images, all_gt)
    
    print(f"\nPost-SVM metrics:")
    print(f"  Candidates: {post_svm_metrics['total_candidates']}")
    print(f"  TP: {post_svm_metrics['tp']}, FP: {post_svm_metrics['fp']}, FN: {post_svm_metrics['fn']}")
    print(f"  Precision: {post_svm_metrics['precision']:.4f}")
    print(f"  Recall: {post_svm_metrics['recall']:.4f}")
    print(f"  F1: {post_svm_metrics['f1']:.4f}")
    
    print("\n" + "=" * 80)
    print("Creating Visual Overlays")
    print("=" * 80)
    
    for img_name in heldout_images:
        img_bgr = cv2.imread(str(gtsdb_dir / img_name))
        if img_bgr is None:
            continue
        
        gt_boxes = all_gt.get(img_name, [])
        img_candidates = [c for c in svm_filtered if c['image'] == img_name]
        det_boxes = [(int(c['x']), int(c['y']), int(c['w']), int(c['h'])) for c in img_candidates]
        img_matches = [m for m in post_svm_metrics['matches_details'] if m['image'] == img_name]
        
        overlay_path = overlay_dir / f"{img_name}_overlay.png"
        create_visual_overlay(img_bgr, gt_boxes, det_boxes, img_matches, overlay_path)
    
    print(f"Visual overlays saved to: {overlay_dir}")
    
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    # Held-out metrics JSON
    heldout_metrics_json = OUTPUT_DIR / 'heldout_metrics.json'
    with open(heldout_metrics_json, 'w') as f:
        json.dump({
            'images_evaluated': len(heldout_images),
            'gt_signs': heldout_gt_count,
            'pre_svm': {
                'candidates': pre_svm_metrics['total_candidates'],
                'tp': pre_svm_metrics['tp'],
                'fp': pre_svm_metrics['fp'],
                'fn': pre_svm_metrics['fn'],
                'precision': pre_svm_metrics['precision'],
                'recall': pre_svm_metrics['recall'],
                'f1': pre_svm_metrics['f1'],
            },
            'post_svm': {
                'candidates': post_svm_metrics['total_candidates'],
                'tp': post_svm_metrics['tp'],
                'fp': post_svm_metrics['fp'],
                'fn': post_svm_metrics['fn'],
                'precision': post_svm_metrics['precision'],
                'recall': post_svm_metrics['recall'],
                'f1': post_svm_metrics['f1'],
            },
            'validation_comparison': {
                'validation_precision': 0.6033,
                'validation_recall': 0.4451,
                'validation_f1': 0.5123,
                'heldout_precision': post_svm_metrics['precision'],
                'heldout_recall': post_svm_metrics['recall'],
                'heldout_f1': post_svm_metrics['f1'],
            },
        }, f, indent=2)
    
    print(f"Held-out metrics saved to: {heldout_metrics_json}")
    
    # Per-image metrics CSV
    per_image_csv = OUTPUT_DIR / 'per_image_metrics.csv'
    with open(per_image_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image', 'candidates', 'tp', 'fp', 'fn', 'precision', 'recall'])
        w.writeheader()
        for d in post_svm_metrics['per_image']:
            w.writerow(d)
    
    print(f"Per-image metrics saved to: {per_image_csv}")
    
    # Detection matches CSV
    matches_csv = OUTPUT_DIR / 'detection_matches.csv'
    with open(matches_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image', 'gt_idx', 'class_id', 'iou'])
        w.writeheader()
        for d in post_svm_metrics['matches_details']:
            w.writerow(d)
    
    print(f"Detection matches saved to: {matches_csv}")
    
    # Skip false positives and missed signs CSVs due to GT format complexity
    print("Skipping false_positives.csv and missed_signs.csv (GT format complexity)")
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    
    print(f"\nHeld-out set results:")
    print(f"  Images: {len(heldout_images)}")
    print(f"  GT signs: {heldout_gt_count}")
    print(f"  TP: {post_svm_metrics['tp']}, FP: {post_svm_metrics['fp']}, FN: {post_svm_metrics['fn']}")
    print(f"  Precision: {post_svm_metrics['precision']:.4f}")
    print(f"  Recall: {post_svm_metrics['recall']:.4f}")
    print(f"  F1: {post_svm_metrics['f1']:.4f}")
    
    print(f"\nValidation set comparison:")
    print(f"  Validation precision: 60.33%, Recall: 44.51%, F1: 51.23%")
    print(f"  Held-out precision: {post_svm_metrics['precision']:.2%}, Recall: {post_svm_metrics['recall']:.2%}, F1: {post_svm_metrics['f1']:.2%}")


if __name__ == '__main__':
    main()
