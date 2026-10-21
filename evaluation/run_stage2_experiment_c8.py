"""Stage 2 Experiment C8: LAB Threshold Relaxation

This experiment tests the effect of relaxing LAB thresholds:
- LAB red a* threshold: 140 -> 130
- LAB blue b* threshold: 115 -> 155
- LAB yellow b* threshold: 140 (unchanged)

Baseline (C3 + SVM): 116 candidates, TP=67, FP=49, FN=97, precision=57.76%, recall=40.85%, F1=47.86%

Acceptance Criteria:
- Recall improvement >= 3 percentage points (>= 43.85%)
- Precision loss <= 10 percentage points (>= 47.76%)
- F1 improvement >= 1 percentage point (>= 48.86%)

Output directory: experiments_phase_n1/stage2_experiment_c8_lab_relaxation/
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
OUTPUT_DIR = Path('experiments_phase_n1/stage2_experiment_c8_lab_relaxation')
MODEL_DIR = Path('experiments_phase_n1/hog_svm_v1')
SVM_THRESHOLD = 0.2
C6_OUTPUT_DIR = Path('experiments_phase_n1/c6_area_filter_feasibility')

# C8 LAB threshold changes
LAB_RED_A_STAR_C8 = 130  # Changed from 140
LAB_BLUE_B_STAR_C8 = 155  # Changed from 115
LAB_YELLOW_B_STAR_C8 = 140  # Unchanged

# C3 AR deviation thresholds (unchanged)
M2D_AR_DEV_THRESHOLD = 0.25
M3D_AR_DEV_THRESHOLD = 0.30


def load_validation_images():
    """Load validation image list from split manifest."""
    manifest_path = Path('experiments_phase_n1/data_preparation/split_manifest.csv')
    val_images = []
    
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == 'val':
                val_images.append(row['image_name'])
    
    return sorted(val_images)


def load_36_difficult_signs():
    """Load the 36 no-candidate GT signs from C6 feasibility analysis."""
    csv_path = C6_OUTPUT_DIR / 'missed_sign_size_analysis.csv'
    difficult_signs = []
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            difficult_signs.append({
                'image': row['image'],
                'gt_idx': int(row['gt_idx']),
                'class_id': int(row['class_id']),
                'x': int(row['gt_x']),
                'y': int(row['gt_y']),
                'w': int(row['gt_w']),
                'h': int(row['gt_h']),
                'area': int(row['gt_area']),
            })
    
    return difficult_signs


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


def measure_mask_coverage(mask: np.ndarray, gt_box: Tuple[int, int, int, int], img_h: int, img_w: int) -> float:
    """Measure coverage of mask within GT box."""
    x, y, w, h = gt_box
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    gt_mask_crop = mask[y1:y2, x1:x2]
    gt_area = (x2 - x1) * (y2 - y1)
    
    if gt_area == 0:
        return 0.0
    
    coverage = np.sum(gt_mask_crop > 0) / gt_area
    return coverage


def analyze_36_difficult_signs(
    cfg: PipelineConfig,
    difficult_signs: List[Dict],
    gtsdb_dir: Path,
) -> List[Dict[str, Any]]:
    """Analyze mask coverage for the 36 difficult signs."""
    results = []
    
    for gt_sign in difficult_signs:
        img_name = gt_sign['image']
        img_bgr = cv2.imread(str(gtsdb_dir / img_name))
        
        if img_bgr is None:
            continue
        
        img_h, img_w = img_bgr.shape[:2]
        gt_box = (gt_sign['x'], gt_sign['y'], gt_sign['w'], gt_sign['h'])
        
        # Stage 1 enhancement
        s1 = run_stage1(img_bgr, cfg.stage1)
        enhanced = s1['enhanced_image']
        
        # Run segmentation
        color_masks = segment_traffic_sign_colors(enhanced, cfg.stage2)
        
        # Measure coverage
        hsv_red_coverage = measure_mask_coverage(color_masks['hsv_red'], gt_box, img_h, img_w)
        hsv_blue_coverage = measure_mask_coverage(color_masks['hsv_blue'], gt_box, img_h, img_w)
        hsv_yellow_coverage = measure_mask_coverage(color_masks['hsv_yellow'], gt_box, img_h, img_w)
        
        lab_red_coverage = measure_mask_coverage(color_masks['lab_red'], gt_box, img_h, img_w)
        lab_blue_coverage = measure_mask_coverage(color_masks['lab_blue'], gt_box, img_h, img_w)
        lab_yellow_coverage = measure_mask_coverage(color_masks['lab_yellow'], gt_box, img_h, img_w)
        
        fused_red_coverage = measure_mask_coverage(color_masks['mask_red'], gt_box, img_h, img_w)
        fused_blue_coverage = measure_mask_coverage(color_masks['mask_blue'], gt_box, img_h, img_w)
        fused_yellow_coverage = measure_mask_coverage(color_masks['mask_yellow'], gt_box, img_h, img_w)
        
        combined_coverage = measure_mask_coverage(color_masks['mask_combined'], gt_box, img_h, img_w)
        
        results.append({
            'image': img_name,
            'gt_idx': gt_sign['gt_idx'],
            'class_id': gt_sign['class_id'],
            'hsv_red_coverage': hsv_red_coverage,
            'hsv_blue_coverage': hsv_blue_coverage,
            'hsv_yellow_coverage': hsv_yellow_coverage,
            'lab_red_coverage': lab_red_coverage,
            'lab_blue_coverage': lab_blue_coverage,
            'lab_yellow_coverage': lab_yellow_coverage,
            'fused_red_coverage': fused_red_coverage,
            'fused_blue_coverage': fused_blue_coverage,
            'fused_yellow_coverage': fused_yellow_coverage,
            'combined_coverage': combined_coverage,
        })
    
    return results


def main():
    root = Path('.').resolve()
    ds_cfg = DatasetConfig()
    gtsdb_dir = root / ds_cfg.gtsdb_dir
    gt_file = root / ds_cfg.gtsdb_gt_file
    
    if OUTPUT_DIR.exists():
        print(f"ERROR: Output directory already exists: {OUTPUT_DIR}")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Stage 2 Experiment C8: LAB Threshold Relaxation")
    print("=" * 80)
    
    val_images = load_validation_images()
    print(f"\nValidation images: {len(val_images)}")
    print(f"Range: {val_images[0]} to {val_images[-1]}")
    
    all_gt, _ = load_gtsdb_annotations(gt_file, gtsdb_dir)
    val_gt_count = sum(len(all_gt.get(img, [])) for img in val_images)
    print(f"Total GT boxes in validation: {val_gt_count}")
    
    difficult_signs = load_36_difficult_signs()
    print(f"Difficult signs to analyze: {len(difficult_signs)}")
    
    # Baseline configuration (C3 with LAB defaults)
    cfg_baseline = PipelineConfig()
    cfg_baseline.stage2.independent_color_morphology = True
    cfg_baseline.stage2.morph_kernel_size = 3
    cfg_baseline.stage2.morph_close_iterations = 3
    cfg_baseline.stage2.morph_open_iterations = 1
    cfg_baseline.stage2.filter9_mode = 'alternative'
    cfg_baseline.stage2.filter11_max_area = 8000
    cfg_baseline.stage2.nms_iou_threshold = 0.40
    cfg_baseline.stage2.hsv_blue_lower = (95, 70, 40)
    cfg_baseline.stage2.hsv_blue_upper = (130, 255, 255)
    cfg_baseline.stage2.lab_blue_lower = (0, 0, 0)
    cfg_baseline.stage2.lab_blue_upper = (255, 255, 125)
    cfg_baseline.stage2.hsv_red_lower1 = (0, 70, 40)
    cfg_baseline.stage2.hsv_red_lower2 = (155, 70, 40)
    cfg_baseline.stage2.hsv_yellow_lower = (14, 70, 50)
    
    print("\n" + "=" * 80)
    print("STEP 1: Baseline Reproduction (C3 with LAB defaults)")
    print("=" * 80)
    
    baseline_survivors, baseline_all = run_m3d_with_config(
        cfg_baseline, val_images, all_gt, gtsdb_dir, 
        m2d_ar_thresh=M2D_AR_DEV_THRESHOLD, m3d_ar_thresh=M3D_AR_DEV_THRESHOLD
    )
    baseline_metrics = evaluate_results(baseline_survivors, val_images, all_gt)
    
    print(f"\nBaseline (C3 + LAB defaults):")
    print(f"  Candidates: {baseline_metrics['total_candidates']}")
    print(f"  TP: {baseline_metrics['tp']}, FP: {baseline_metrics['fp']}, FN: {baseline_metrics['fn']}")
    print(f"  Precision: {baseline_metrics['precision']:.4f}")
    print(f"  Recall: {baseline_metrics['recall']:.4f}")
    print(f"  F1: {baseline_metrics['f1']:.4f}")
    
    expected_candidates = 2250
    expected_tp = 69
    expected_fp = 2181
    expected_fn = 95
    
    if baseline_metrics['total_candidates'] != expected_candidates:
        print(f"WARNING: Candidate count mismatch: {baseline_metrics['total_candidates']} vs {expected_candidates}")
    if baseline_metrics['tp'] != expected_tp:
        print(f"WARNING: TP mismatch: {baseline_metrics['tp']} vs {expected_tp}")
    
    # C8 configuration (LAB relaxed)
    cfg_c8 = PipelineConfig()
    cfg_c8.stage2.independent_color_morphology = True
    cfg_c8.stage2.morph_kernel_size = 3
    cfg_c8.stage2.morph_close_iterations = 3
    cfg_c8.stage2.morph_open_iterations = 1
    cfg_c8.stage2.filter9_mode = 'alternative'
    cfg_c8.stage2.filter11_max_area = 8000
    cfg_c8.stage2.nms_iou_threshold = 0.40
    cfg_c8.stage2.hsv_blue_lower = (95, 70, 40)
    cfg_c8.stage2.hsv_blue_upper = (130, 255, 255)
    cfg_c8.stage2.lab_blue_lower = (0, 0, 0)
    cfg_c8.stage2.lab_blue_upper = (255, 255, 125)
    cfg_c8.stage2.hsv_red_lower1 = (0, 70, 40)
    cfg_c8.stage2.hsv_red_lower2 = (155, 70, 40)
    cfg_c8.stage2.hsv_yellow_lower = (14, 70, 50)
    
    # C8 LAB threshold changes
    cfg_c8.stage2.lab_red_lower = (20, LAB_RED_A_STAR_C8, 100)
    cfg_c8.stage2.lab_red_upper = (255, 255, 255)
    cfg_c8.stage2.lab_blue_lower = (20, 60, 20)
    cfg_c8.stage2.lab_blue_upper = (230, 145, LAB_BLUE_B_STAR_C8)
    cfg_c8.stage2.lab_yellow_lower = (40, 100, LAB_YELLOW_B_STAR_C8)
    cfg_c8.stage2.lab_yellow_upper = (255, 160, 255)
    
    print("\n" + "=" * 80)
    print("STEP 2: C8 Evaluation (LAB relaxed)")
    print("=" * 80)
    
    c8_survivors, c8_all = run_m3d_with_config(
        cfg_c8, val_images, all_gt, gtsdb_dir,
        m2d_ar_thresh=M2D_AR_DEV_THRESHOLD, m3d_ar_thresh=M3D_AR_DEV_THRESHOLD
    )
    c8_metrics = evaluate_results(c8_survivors, val_images, all_gt)
    
    print(f"\nC8 (LAB relaxed):")
    print(f"  Candidates: {c8_metrics['total_candidates']}")
    print(f"  TP: {c8_metrics['tp']}, FP: {c8_metrics['fp']}, FN: {c8_metrics['fn']}")
    print(f"  Precision: {c8_metrics['precision']:.4f}")
    print(f"  Recall: {c8_metrics['recall']:.4f}")
    print(f"  F1: {c8_metrics['f1']:.4f}")
    
    cand_change = c8_metrics['total_candidates'] - baseline_metrics['total_candidates']
    tp_change = c8_metrics['tp'] - baseline_metrics['tp']
    fp_change = c8_metrics['fp'] - baseline_metrics['fp']
    fn_change = c8_metrics['fn'] - baseline_metrics['fn']
    
    print(f"\nChanges from baseline:")
    print(f"  Candidates: {cand_change} ({cand_change/baseline_metrics['total_candidates']*100:.1f}%)")
    print(f"  TP: {tp_change} ({tp_change/baseline_metrics['tp']*100:.1f}%)")
    print(f"  FP: {fp_change} ({fp_change/baseline_metrics['fp']*100:.1f}%)")
    print(f"  FN: {fn_change}")
    
    # Analyze lost/gained signs
    baseline_keys = baseline_metrics['matches_keys']
    c8_keys = c8_metrics['matches_keys']
    lost_signs = baseline_keys - c8_keys
    gained_signs = c8_keys - baseline_keys
    
    print(f"\nSigns lost: {len(lost_signs)}")
    print(f"Signs gained: {len(gained_signs)}")
    
    print("\n" + "=" * 80)
    print("STEP 3: Analyze 36 Difficult Signs")
    print("=" * 80)
    
    baseline_difficult = analyze_36_difficult_signs(cfg_baseline, difficult_signs, gtsdb_dir)
    c8_difficult = analyze_36_difficult_signs(cfg_c8, difficult_signs, gtsdb_dir)
    
    print(f"\nBaseline coverage for 36 difficult signs:")
    baseline_avg_combined = np.mean([d['combined_coverage'] for d in baseline_difficult])
    print(f"  Average combined coverage: {baseline_avg_combined:.4f}")
    
    print(f"\nC8 coverage for 36 difficult signs:")
    c8_avg_combined = np.mean([d['combined_coverage'] for d in c8_difficult])
    print(f"  Average combined coverage: {c8_avg_combined:.4f}")
    
    print(f"  Improvement: {c8_avg_combined - baseline_avg_combined:.4f}")
    
    print("\n" + "=" * 80)
    print("STEP 4: Full Pipeline with SVM")
    print("=" * 80)
    
    try:
        svm = joblib.load(MODEL_DIR / 'svm_model.joblib')
        scaler = joblib.load(MODEL_DIR / 'scaler.joblib')
        
        with open(MODEL_DIR / 'hog_config.json', 'r') as f:
            hog_config = json.load(f)
        
        print("SVM model loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load SVM model: {e}")
        return
    
    # Baseline with SVM
    baseline_svm_filtered = []
    for img_name in val_images:
        img_bgr = cv2.imread(str(gtsdb_dir / img_name))
        if img_bgr is None:
            continue
        img_candidates = [c for c in baseline_survivors if c['image'] == img_name]
        filtered = svm_filter_candidates(img_candidates, img_bgr, svm, scaler, hog_config, SVM_THRESHOLD)
        baseline_svm_filtered.extend(filtered)
    
    baseline_svm_metrics = evaluate_results(baseline_svm_filtered, val_images, all_gt)
    
    print(f"\nBaseline + SVM:")
    print(f"  Candidates: {baseline_svm_metrics['total_candidates']}")
    print(f"  TP: {baseline_svm_metrics['tp']}, FP: {baseline_svm_metrics['fp']}, FN: {baseline_svm_metrics['fn']}")
    print(f"  Precision: {baseline_svm_metrics['precision']:.4f}")
    print(f"  Recall: {baseline_svm_metrics['recall']:.4f}")
    print(f"  F1: {baseline_svm_metrics['f1']:.4f}")
    
    # C8 with SVM
    c8_svm_filtered = []
    for img_name in val_images:
        img_bgr = cv2.imread(str(gtsdb_dir / img_name))
        if img_bgr is None:
            continue
        img_candidates = [c for c in c8_survivors if c['image'] == img_name]
        filtered = svm_filter_candidates(img_candidates, img_bgr, svm, scaler, hog_config, SVM_THRESHOLD)
        c8_svm_filtered.extend(filtered)
    
    c8_svm_metrics = evaluate_results(c8_svm_filtered, val_images, all_gt)
    
    print(f"\nC8 + SVM:")
    print(f"  Candidates: {c8_svm_metrics['total_candidates']}")
    print(f"  TP: {c8_svm_metrics['tp']}, FP: {c8_svm_metrics['fp']}, FN: {c8_svm_metrics['fn']}")
    print(f"  Precision: {c8_svm_metrics['precision']:.4f}")
    print(f"  Recall: {c8_svm_metrics['recall']:.4f}")
    print(f"  F1: {c8_svm_metrics['f1']:.4f}")
    
    svm_cand_change = c8_svm_metrics['total_candidates'] - baseline_svm_metrics['total_candidates']
    svm_tp_change = c8_svm_metrics['tp'] - baseline_svm_metrics['tp']
    svm_fp_change = c8_svm_metrics['fp'] - baseline_svm_metrics['fp']
    
    print(f"\nFull pipeline changes from baseline:")
    print(f"  Candidates: {svm_cand_change}")
    print(f"  TP: {svm_tp_change}")
    print(f"  FP: {svm_fp_change}")
    
    # Analyze SVM changes
    baseline_svm_keys = baseline_svm_metrics['matches_keys']
    c8_svm_keys = c8_svm_metrics['matches_keys']
    svm_lost_signs = baseline_svm_keys - c8_svm_keys
    svm_gained_signs = c8_svm_keys - baseline_svm_keys
    
    print(f"\nSVM signs lost: {len(svm_lost_signs)}")
    print(f"SVM signs gained: {len(svm_gained_signs)}")
    
    print("\n" + "=" * 80)
    print("STEP 5: Evaluate Acceptance Criteria")
    print("=" * 80)
    
    baseline_precision = baseline_svm_metrics['precision']
    baseline_recall = baseline_svm_metrics['recall']
    baseline_f1 = baseline_svm_metrics['f1']
    
    c8_precision = c8_svm_metrics['precision']
    c8_recall = c8_svm_metrics['recall']
    c8_f1 = c8_svm_metrics['f1']
    
    recall_improvement = (c8_recall - baseline_recall) * 100
    precision_loss = (baseline_precision - c8_precision) * 100
    f1_gain = (c8_f1 - baseline_f1) * 100
    
    print(f"\nRecall improvement: {recall_improvement:.2f} percentage points")
    print(f"Precision loss: {precision_loss:.2f} percentage points")
    print(f"F1 gain: {f1_gain:.2f} percentage points")
    
    criteria_pass = {
        'recall_target': c8_recall >= 0.4385,
        'precision_target': c8_precision >= 0.4776,
        'f1_target': c8_f1 >= 0.4886,
    }
    
    print(f"\nAcceptance Criteria:")
    print(f"  Recall >= 43.85%: {'PASS' if criteria_pass['recall_target'] else 'FAIL'} ({c8_recall*100:.2f}% vs 43.85%)")
    print(f"  Precision >= 47.76%: {'PASS' if criteria_pass['precision_target'] else 'FAIL'} ({c8_precision*100:.2f}% vs 47.76%)")
    print(f"  F1 >= 48.86%: {'PASS' if criteria_pass['f1_target'] else 'FAIL'} ({c8_f1*100:.2f}% vs 48.86%)")
    
    all_pass = all(criteria_pass.values())
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    
    print("\n" + "=" * 80)
    print("STEP 6: Saving Results")
    print("=" * 80)
    
    # Comparison CSV
    comparison_csv = OUTPUT_DIR / 'metrics_comparison.csv'
    with open(comparison_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['config', 'candidates', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])
        w.writeheader()
        w.writerows([
            {
                'config': 'baseline_c3_lab_defaults',
                'candidates': baseline_metrics['total_candidates'],
                'tp': baseline_metrics['tp'],
                'fp': baseline_metrics['fp'],
                'fn': baseline_metrics['fn'],
                'precision': f"{baseline_metrics['precision']:.4f}",
                'recall': f"{baseline_metrics['recall']:.4f}",
                'f1': f"{baseline_metrics['f1']:.4f}",
            },
            {
                'config': 'c8_lab_relaxed',
                'candidates': c8_metrics['total_candidates'],
                'tp': c8_metrics['tp'],
                'fp': c8_metrics['fp'],
                'fn': c8_metrics['fn'],
                'precision': f"{c8_metrics['precision']:.4f}",
                'recall': f"{c8_metrics['recall']:.4f}",
                'f1': f"{c8_metrics['f1']:.4f}",
            },
            {
                'config': 'baseline_c3_lab_defaults_svm',
                'candidates': baseline_svm_metrics['total_candidates'],
                'tp': baseline_svm_metrics['tp'],
                'fp': baseline_svm_metrics['fp'],
                'fn': baseline_svm_metrics['fn'],
                'precision': f"{baseline_svm_metrics['precision']:.4f}",
                'recall': f"{baseline_svm_metrics['recall']:.4f}",
                'f1': f"{baseline_svm_metrics['f1']:.4f}",
            },
            {
                'config': 'c8_lab_relaxed_svm',
                'candidates': c8_svm_metrics['total_candidates'],
                'tp': c8_svm_metrics['tp'],
                'fp': c8_svm_metrics['fp'],
                'fn': c8_svm_metrics['fn'],
                'precision': f"{c8_svm_metrics['precision']:.4f}",
                'recall': f"{c8_svm_metrics['recall']:.4f}",
                'f1': f"{c8_svm_metrics['f1']:.4f}",
            },
        ])
    
    # Per-image metrics
    per_image_csv = OUTPUT_DIR / 'per_image_metrics.csv'
    with open(per_image_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image', 'baseline_candidates', 'c8_candidates', 'baseline_tp', 'c8_tp', 'baseline_fp', 'c8_fp', 'baseline_fn', 'c8_fn'])
        w.writeheader()
        for img_name in val_images:
            baseline_img_data = next((d for d in baseline_metrics['per_image'] if d['image'] == img_name), {'candidates': 0, 'tp': 0, 'fp': 0, 'fn': 0})
            c8_img_data = next((d for d in c8_metrics['per_image'] if d['image'] == img_name), {'candidates': 0, 'tp': 0, 'fp': 0, 'fn': 0})
            w.writerow({
                'image': img_name,
                'baseline_candidates': baseline_img_data['candidates'],
                'c8_candidates': c8_img_data['candidates'],
                'baseline_tp': baseline_img_data['tp'],
                'c8_tp': c8_img_data['tp'],
                'baseline_fp': baseline_img_data['fp'],
                'c8_fp': c8_img_data['fp'],
                'baseline_fn': baseline_img_data['fn'],
                'c8_fn': c8_img_data['fn'],
            })
    
    # Recovered and lost signs
    signs_csv = OUTPUT_DIR / 'recovered_and_lost_signs.csv'
    with open(signs_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['config', 'sign', 'status'])
        w.writeheader()
        for sign in lost_signs:
            w.writerow({'config': 'c8_vs_baseline_svm', 'sign': sign, 'status': 'lost'})
        for sign in gained_signs:
            w.writerow({'config': 'c8_vs_baseline_svm', 'sign': sign, 'status': 'recovered'})
    
    # Mask coverage comparison for 36 difficult signs
    mask_coverage_csv = OUTPUT_DIR / 'mask_coverage_comparison.csv'
    with open(mask_coverage_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'image', 'gt_idx', 'class_id',
            'baseline_combined_coverage', 'c8_combined_coverage',
            'baseline_lab_red_coverage', 'c8_lab_red_coverage',
            'baseline_lab_blue_coverage', 'c8_lab_blue_coverage',
            'baseline_lab_yellow_coverage', 'c8_lab_yellow_coverage',
        ])
        w.writeheader()
        for b, c in zip(baseline_difficult, c8_difficult):
            w.writerow({
                'image': b['image'],
                'gt_idx': b['gt_idx'],
                'class_id': b['class_id'],
                'baseline_combined_coverage': b['combined_coverage'],
                'c8_combined_coverage': c['combined_coverage'],
                'baseline_lab_red_coverage': b['lab_red_coverage'],
                'c8_lab_red_coverage': c['lab_red_coverage'],
                'baseline_lab_blue_coverage': b['lab_blue_coverage'],
                'c8_lab_blue_coverage': c['lab_blue_coverage'],
                'baseline_lab_yellow_coverage': b['lab_yellow_coverage'],
                'c8_lab_yellow_coverage': c['lab_yellow_coverage'],
            })
    
    # Summary JSON
    summary_json = OUTPUT_DIR / 'summary.json'
    with open(summary_json, 'w') as f:
        json.dump({
            'config': {
                'lab_red_a_star': LAB_RED_A_STAR_C8,
                'lab_blue_b_star': LAB_BLUE_B_STAR_C8,
                'lab_yellow_b_star': LAB_YELLOW_B_STAR_C8,
            },
            'baseline_pre_svm': {
                'candidates': baseline_metrics['total_candidates'],
                'tp': baseline_metrics['tp'],
                'fp': baseline_metrics['fp'],
                'fn': baseline_metrics['fn'],
                'precision': baseline_metrics['precision'],
                'recall': baseline_metrics['recall'],
                'f1': baseline_metrics['f1'],
            },
            'c8_pre_svm': {
                'candidates': c8_metrics['total_candidates'],
                'tp': c8_metrics['tp'],
                'fp': c8_metrics['fp'],
                'fn': c8_metrics['fn'],
                'precision': c8_metrics['precision'],
                'recall': c8_metrics['recall'],
                'f1': c8_metrics['f1'],
            },
            'baseline_post_svm': {
                'candidates': baseline_svm_metrics['total_candidates'],
                'tp': baseline_svm_metrics['tp'],
                'fp': baseline_svm_metrics['fp'],
                'fn': baseline_svm_metrics['fn'],
                'precision': baseline_svm_metrics['precision'],
                'recall': baseline_svm_metrics['recall'],
                'f1': baseline_svm_metrics['f1'],
            },
            'c8_post_svm': {
                'candidates': c8_svm_metrics['total_candidates'],
                'tp': c8_svm_metrics['tp'],
                'fp': c8_svm_metrics['fp'],
                'fn': c8_svm_metrics['fn'],
                'precision': c8_svm_metrics['precision'],
                'recall': c8_svm_metrics['recall'],
                'f1': c8_svm_metrics['f1'],
            },
            'acceptance_criteria': {
                'recall_target': 0.4385,
                'recall_measured': c8_recall,
                'recall_pass': criteria_pass['recall_target'],
                'precision_target': 0.4776,
                'precision_measured': c8_precision,
                'precision_pass': criteria_pass['precision_target'],
                'f1_target': 0.4886,
                'f1_measured': c8_f1,
                'f1_pass': criteria_pass['f1_target'],
                'all_pass': all_pass,
            },
            'signs_lost_count': len(svm_lost_signs),
            'signs_gained_count': len(svm_gained_signs),
            'difficult_signs_coverage_improvement': c8_avg_combined - baseline_avg_combined,
        }, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    print("\n" + "=" * 80)
    print("Experiment complete!")
    print("=" * 80)
    
    print(f"\nRecommendation: {'KEEP LAB relaxed' if all_pass else 'REVERT to LAB defaults'}")


if __name__ == '__main__':
    main()
