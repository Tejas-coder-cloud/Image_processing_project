"""Phase M — Experiment M2: Targeted Colour-Branch Foliage Suppression.

Baseline:
- Phase M1: K2 + A3 + B2 + C2 score >= 0.200 + vertical corridor [0.25, 0.70]
- Verified M1: candidates=1688, TP=17, FP=1671, FN=12, Recall=58.62%, Precision=1.01%

Experimental Conditions tested against M1:
- M2_A: Color-branch Contour Solidity Filtering (solidity >= 0.20)
- M2_B: Color-branch Bilateral Edge Symmetry Gate (sym >= 0.12)
- M2_C: Color-branch Aspect Ratio Deviation Gate (ar_deviation <= 0.42, i.e. 0.70 <= AR <= 1.42)
- M2_D: Combined Multi-Criterion Targeted Foliage Filter (A + B + C on color branch)

All conditions preserve:
- Edge candidates untouched
- Pre-filtering steps (enhancement, thresholds, morphology, NMS, scoring, M1 corridor)
- Strict greedy bipartite matching at IoU >= 0.50

Outputs saved under experiments_phase_m/:
- phase_m2_comparison_summary.csv
- phase_m2_per_image.csv
- phase_m2_tp_audit.csv
- phase_m2_report.md
- Overlays in experiments_phase_m/m2_overlays/
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from src.config import PipelineConfig, DatasetConfig
from src.pipeline import run_stage1
from src.stage2_segmentation.thresholding import segment_traffic_sign_colors
from src.run_phase_l_expc import (
    process_all_images, evaluate_one_to_one
)


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


def compute_color_mask_solidity(crop_mask: np.ndarray) -> float:
    """Compute solidity of the largest contour in the color mask crop."""
    if crop_mask.size == 0 or np.count_nonzero(crop_mask) == 0:
        return 0.0
    cnts, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c_max = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c_max)
    hull = cv2.convexHull(c_max)
    hull_area = cv2.contourArea(hull)
    return float(area / max(1.0, hull_area))


def run_phase_m2():
    root = Path('.').resolve()
    ds_cfg = DatasetConfig()
    gtsdb_dir = root / ds_cfg.gtsdb_dir
    gt_file = root / ds_cfg.gtsdb_gt_file
    out_dir = root / 'experiments_phase_m'
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / 'm2_overlays'
    overlay_dir.mkdir(parents=True, exist_ok=True)

    sample_images = [f'{i:05d}.ppm' for i in range(20)]
    all_gt, _ = load_gtsdb_annotations(gt_file, gtsdb_dir)

    cfg = PipelineConfig()
    cfg.stage2.independent_color_morphology = True
    cfg.stage2.morph_kernel_size = 3
    cfg.stage2.morph_close_iterations = 3
    cfg.stage2.morph_open_iterations = 1
    cfg.stage2.filter9_mode = 'alternative'
    cfg.stage2.filter11_max_area = 8000
    cfg.stage2.nms_iou_threshold = 0.40
    cfg.stage2.hsv_blue_lower = (100, 30, 30)
    cfg.stage2.hsv_blue_upper = (130, 255, 255)
    cfg.stage2.lab_blue_lower = (0, 0, 0)
    cfg.stage2.lab_blue_upper = (255, 255, 125)

    print("Phase M2: Extracting base candidates and computing features...")
    all_records = process_all_images(cfg, sample_images, all_gt, gtsdb_dir)

    # 1. Apply Phase L C2 score >= 0.200
    c2_survivors = [r for r in all_records if float(r['score_c2']) >= 0.200]

    # 2. Apply Phase M1 Vertical Corridor [0.25, 0.70]
    m1_survivors = [r for r in c2_survivors if 0.25 <= float(r['center_y_ratio']) <= 0.70]
    print(f"M1 Baseline candidate pool: {len(m1_survivors)}")

    # Precompute color mask solidity for each color candidate
    img_cache = {}
    sol_map = {}

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
            x, y, w, h = int(r['x']), int(r['y']), int(r['w']), int(r['h'])
            crop_m = mask_comb[y:y+h, x:x+w]
            sol = compute_color_mask_solidity(crop_m)
            sol_map[(im_name, x, y, w, h)] = sol
            r['solidity'] = sol
        else:
            r['solidity'] = 1.0  # N/A for edge candidates

    # Conditions definitions:
    # A candidate is accepted if: source == 'edge' OR filter_predicate(color_candidate)
    conditions = [
        ("M1_Baseline", lambda r: True),
        ("M2_A_Solidity_0.20", lambda r: r['source'] == 'edge' or r['solidity'] >= 0.20),
        ("M2_B_Symmetry_0.12", lambda r: r['source'] == 'edge' or float(r['edge_symmetry']) >= 0.12),
        ("M2_C_Aspect_0.42", lambda r: r['source'] == 'edge' or float(r['ar_deviation']) <= 0.42),
        ("M2_D_Combined_ABC", lambda r: r['source'] == 'edge' or (
            r['solidity'] >= 0.20 and float(r['edge_symmetry']) >= 0.12 and float(r['ar_deviation']) <= 0.42
        )),
    ]

    c12_signs = {"00008.ppm:GT#1(cls12)", "00010.ppm:GT#0(cls12)", "00016.ppm:GT#0(cls12)", "00019.ppm:GT#0(cls12)"}
    cls21_ref = "00014.ppm:GT#0(cls21)"

    cond_results = []
    all_cond_gt_status = {}
    per_image_summary = []

    print("\n" + "="*115)
    print(f"{'Condition':<22} | {'Cands':>5} | {'TP':>2} | {'FP':>5} | {'FN':>2} | {'Prec':>7} | {'Rec':>6} | {'F1':>6} | {'dTP':>3} | {'dFP':>5} | {'C12':>4} | {'00014_cls21'}")
    print("="*115)

    for cond_name, filter_fn in conditions:
        surviving = [r for r in m1_survivors if filter_fn(r)]

        by_img: Dict[str, List[Dict]] = {}
        for r in surviving:
            by_img.setdefault(r['image'], []).append(r)

        c_tp = c_fp = c_fn = 0
        matches_keys = set()
        matched_details = []

        for img_name in sample_images:
            recs = by_img.get(img_name, [])
            det_boxes = [(int(r['x']), int(r['y']), int(r['w']), int(r['h'])) for r in recs]
            gt_boxes = all_gt.get(img_name, [])
            ev = evaluate_one_to_one(det_boxes, gt_boxes, 0.50)
            c_tp += ev['tp']
            c_fp += ev['fp']
            c_fn += ev['fn']

            matched_map = {m['gt_idx']: m for m in ev['matches']}
            for m in ev['matches']:
                key = f"{img_name}:GT#{m['gt_idx']}(cls{m['class_id']})"
                matches_keys.add(key)
                matched_details.append({
                    'condition': cond_name, 'image': img_name, 'gt_idx': m['gt_idx'],
                    'class_id': m['class_id'], 'iou': m['iou'], 'det_box': m['det_box']
                })

            per_image_summary.append({
                'condition': cond_name, 'image': img_name,
                'cands': len(recs), 'tp': ev['tp'], 'fp': ev['fp'], 'fn': ev['fn'],
                'precision': f"{ev['precision']:.4f}", 'recall': f"{ev['recall']:.4f}"
            })

            # Overlays for M2_D
            if cond_name == "M2_D_Combined_ABC":
                img_bgr = cv2.imread(str(gtsdb_dir / img_name))
                ov = draw_overlay(img_bgr, det_boxes, gt_boxes, ev, title=f"Phase M2-D: {img_name}")
                cv2.imwrite(str(overlay_dir / f"{Path(img_name).stem}_m2d_overlay.png"), ov)

        assert c_tp + c_fn == 29, f"TP+FN != 29 for {cond_name}"
        assert c_tp + c_fp == len(surviving), f"TP+FP != surviving for {cond_name}"

        prec = c_tp / max(1, c_tp + c_fp)
        rec = c_tp / 29.0
        f1 = 2 * prec * rec / max(1e-8, prec + rec)

        c12_count = len(c12_signs & matches_keys)
        cls21_ok = cls21_ref in matches_keys

        dtp = c_tp - 17
        dfp = c_fp - 1671

        all_cond_gt_status[cond_name] = matches_keys

        cond_results.append({
            'condition': cond_name, 'candidates': len(surviving),
            'tp': c_tp, 'fp': c_fp, 'fn': c_fn,
            'precision': prec, 'recall': rec, 'f1': f1,
            'delta_tp': dtp, 'delta_fp': dfp,
            'class12_kept': f"{c12_count}/4",
            'cls21_00014': cls21_ok
        })

        print(f"{cond_name:<22} | {len(surviving):>5} | {c_tp:>2} | {c_fp:>5} | {c_fn:>2} | {prec:>7.4f} | {rec:>6.4f} | {f1:>6.4f} | {dtp:>+3d} | {dfp:>+5d} | {c12_count:>2}/4 | {'OK' if cls21_ok else 'LOST'}")

    print("="*115)

    # Save summary CSV
    sum_csv = out_dir / 'phase_m2_comparison_summary.csv'
    with open(sum_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cond_results[0].keys()))
        w.writeheader()
        w.writerows(cond_results)
    print(f"\nSaved comparison summary: {sum_csv}")

    # Save per-image CSV
    per_img_csv = out_dir / 'phase_m2_per_image.csv'
    with open(per_img_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_summary[0].keys()))
        w.writeheader()
        w.writerows(per_image_summary)
    print(f"Saved per-image summary: {per_img_csv}")

    # Save TP audit across conditions
    tp_audit_csv = out_dir / 'phase_m2_tp_audit.csv'
    with open(tp_audit_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_gt', 'M1_Baseline', 'M2_A_Solidity', 'M2_B_Symmetry', 'M2_C_Aspect', 'M2_D_Combined'])
        m1_set = all_cond_gt_status['M1_Baseline']
        for key in sorted(m1_set):
            w.writerow([
                key,
                key in all_cond_gt_status['M1_Baseline'],
                key in all_cond_gt_status['M2_A_Solidity_0.20'],
                key in all_cond_gt_status['M2_B_Symmetry_0.12'],
                key in all_cond_gt_status['M2_C_Aspect_0.42'],
                key in all_cond_gt_status['M2_D_Combined_ABC'],
            ])
    print(f"Saved TP audit: {tp_audit_csv}")


if __name__ == '__main__':
    run_phase_m2()
