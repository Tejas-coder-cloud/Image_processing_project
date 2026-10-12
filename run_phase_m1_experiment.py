"""Phase M — Experiment M1: Vertical Spatial Corridor Filtering.

Baseline:
- K2 + A3 + B2 + C2 score threshold 0.200
- Verified baseline: Post-NMS=2635, TP=17, FP=2618, FN=12, Recall=58.62%, Precision=0.65%

Experiment M1:
- Apply vertical candidate-center corridor:
    0.25 <= (y + h/2) / image_height <= 0.70
- Applied post-score / post-NMS, right before evaluation, preserving:
    - color masks
    - edge candidate generation
    - NMS settings
    - C2 scoring formula and threshold (0.200)

Evaluation:
- 20 GTSDB images, 29 GT signs, IoU >= 0.50, greedy one-to-one matching
- Full audit of all 17 TPs before & after M1
- Rejection breakdown by branch, image, and vertical position (sky vs road)
- Validation of invariants: TP + FN = 29, TP + FP = post_filter_count
"""

import csv, os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from src.config import PipelineConfig, DatasetConfig
from evaluate_and_compare import load_gtsdb_annotations, compute_iou_xywh, draw_overlay
from run_phase_l_expc import (
    process_all_images, evaluate_one_to_one
)


def run_phase_m1():
    root = Path('.').resolve()
    ds_cfg = DatasetConfig()
    gtsdb_dir = root / ds_cfg.gtsdb_dir
    gt_file = root / ds_cfg.gtsdb_gt_file
    out_dir = root / 'experiments_phase_m'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'm1_overlays').mkdir(exist_ok=True)

    sample_images = [f'{i:05d}.ppm' for i in range(20)]
    all_gt, _ = load_gtsdb_annotations(gt_file, gtsdb_dir)

    # Reference pipeline config (K2 + A3 + B2)
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

    print("Phase M1: Extracting and scoring candidates across 20 images...")
    # Get all post-NMS candidate records with features & C2 score
    all_records = process_all_images(cfg, sample_images, all_gt, gtsdb_dir)

    # Filter by C2 score threshold 0.200 (exact Phase L baseline)
    c2_thresh = 0.200
    baseline_records = [r for r in all_records if float(r['score_c2']) >= c2_thresh]
    print(f"Baseline (K2+A3+B2 + C2>=0.200) candidate count: {len(baseline_records)}")

    # Group baseline candidates by image for evaluation
    by_img_baseline: Dict[str, List[Dict]] = {}
    for r in baseline_records:
        by_img_baseline.setdefault(r['image'], []).append(r)

    # Baseline one-to-one evaluation
    b_tp = b_fp = b_fn = 0
    b_matches = []
    for img_name in sample_images:
        recs = by_img_baseline.get(img_name, [])
        det_boxes = [(int(r['x']), int(r['y']), int(r['w']), int(r['h'])) for r in recs]
        gt_boxes = all_gt.get(img_name, [])
        ev = evaluate_one_to_one(det_boxes, gt_boxes, 0.50)
        b_tp += ev['tp']
        b_fp += ev['fp']
        b_fn += ev['fn']
        for m in ev['matches']:
            # tag matching record details
            det_b = m['det_box']
            matched_rec = next(r for r in recs if (int(r['x']), int(r['y']), int(r['w']), int(r['h'])) == det_b)
            b_matches.append({
                'image': img_name,
                'gt_idx': m['gt_idx'],
                'class_id': m['class_id'],
                'iou': m['iou'],
                'source': matched_rec['source'],
                'det_box': det_b,
                'center_y_ratio': float(matched_rec['center_y_ratio']),
                'score_c2': float(matched_rec['score_c2']),
            })

    print(f"\n--- Baseline Verification ---")
    print(f"Candidates: {len(baseline_records)}, TP: {b_tp}, FP: {b_fp}, FN: {b_fn}, "
          f"Precision: {b_tp/len(baseline_records):.4f}, Recall: {b_tp/29:.4f}")
    assert b_tp == 17 and b_fp == 2618 and b_fn == 12, "Baseline mismatch!"
    print("Baseline verified exactly (TP=17, FP=2618, FN=12).")

    # -------------------------------------------------------------
    # Apply M1: Vertical Spatial Corridor Constraint [0.25, 0.70]
    # -------------------------------------------------------------
    min_y_ratio = 0.25
    max_y_ratio = 0.70

    m1_surviving: List[Dict] = []
    m1_rejected: List[Dict] = []

    for r in baseline_records:
        y_r = float(r['center_y_ratio'])
        if min_y_ratio <= y_r <= max_y_ratio:
            m1_surviving.append(r)
        else:
            rej_reason = "sky_canopy_high" if y_r < min_y_ratio else "road_surface_low"
            r_copy = dict(r)
            r_copy['m1_rejection_reason'] = rej_reason
            m1_rejected.append(r_copy)

    print(f"\n--- M1 Filtering Statistics ---")
    print(f"Candidates before M1: {len(baseline_records)}")
    print(f"Candidates after M1:  {len(m1_surviving)}")
    print(f"Candidates rejected:  {len(m1_rejected)} ({100*len(m1_rejected)/len(baseline_records):.2f}% reduction)")

    rej_by_reason = {}
    for r in m1_rejected:
        reason = r['m1_rejection_reason']
        rej_by_reason[reason] = rej_by_reason.get(reason, 0) + 1
    print(f"Rejections by region: {rej_by_reason}")

    rej_by_source = {}
    for r in m1_rejected:
        src = r['source']
        rej_by_source[src] = rej_by_source.get(src, 0) + 1
    print(f"Rejections by branch source: {rej_by_source}")

    # Evaluate M1 post-filtering
    by_img_m1: Dict[str, List[Dict]] = {}
    for r in m1_surviving:
        by_img_m1.setdefault(r['image'], []).append(r)

    m1_tp = m1_fp = m1_fn = 0
    m1_matches = []
    per_image_rows = []
    per_gt_rows = []

    for img_name in sample_images:
        recs = by_img_m1.get(img_name, [])
        det_boxes = [(int(r['x']), int(r['y']), int(r['w']), int(r['h'])) for r in recs]
        gt_boxes = all_gt.get(img_name, [])
        ev = evaluate_one_to_one(det_boxes, gt_boxes, 0.50)

        m1_tp += ev['tp']
        m1_fp += ev['fp']
        m1_fn += ev['fn']

        matched_map = {m['gt_idx']: m for m in ev['matches']}
        for m in ev['matches']:
            det_b = m['det_box']
            matched_rec = next(r for r in recs if (int(r['x']), int(r['y']), int(r['w']), int(r['h'])) == det_b)
            m1_matches.append({
                'image': img_name,
                'gt_idx': m['gt_idx'],
                'class_id': m['class_id'],
                'iou': m['iou'],
                'source': matched_rec['source'],
                'det_box': det_b,
                'center_y_ratio': float(matched_rec['center_y_ratio']),
                'score_c2': float(matched_rec['score_c2']),
            })

        for gi, gt in enumerate(gt_boxes):
            is_tp = gi in matched_map
            m_info = matched_map.get(gi)
            per_gt_rows.append({
                'image': img_name,
                'gt_idx': gi,
                'class_id': gt['class_id'],
                'gt_box': f"({gt['x']},{gt['y']},{gt['w']},{gt['h']})",
                'status': 'DETECTED_TP' if is_tp else 'MISSED_FN',
                'matched_det_box': str(m_info['det_box']) if is_tp else 'None',
                'iou': f"{m_info['iou']:.4f}" if is_tp else '0.0000',
            })

        # Count rejections for this image
        img_rej = [r for r in m1_rejected if r['image'] == img_name]
        rej_sky = sum(1 for r in img_rej if r['m1_rejection_reason'] == 'sky_canopy_high')
        rej_road = sum(1 for r in img_rej if r['m1_rejection_reason'] == 'road_surface_low')

        per_image_rows.append({
            'image': img_name,
            'gt_count': len(gt_boxes),
            'baseline_cands': len(by_img_baseline.get(img_name, [])),
            'm1_cands': len(recs),
            'tp': ev['tp'], 'fp': ev['fp'], 'fn': ev['fn'],
            'precision': f"{ev['precision']:.4f}",
            'recall': f"{ev['recall']:.4f}",
            'rejected_total': len(img_rej),
            'rejected_sky': rej_sky,
            'rejected_road': rej_road,
        })

        # Draw overlays
        img_bgr = cv2.imread(str(gtsdb_dir / img_name))
        ov = draw_overlay(img_bgr, det_boxes, gt_boxes, ev, title=f"Phase M1: {img_name}")
        cv2.imwrite(str(out_dir / 'm1_overlays' / f"{Path(img_name).stem}_m1_overlay.png"), ov)

    # Invariants verification
    assert m1_tp + m1_fn == 29, f"TP+FN ({m1_tp+m1_fn}) != 29"
    assert m1_tp + m1_fp == len(m1_surviving), f"TP+FP ({m1_tp+m1_fp}) != surviving count ({len(m1_surviving)})"

    prec = m1_tp / max(1, m1_tp + m1_fp)
    rec = m1_tp / 29.0
    f1 = 2 * prec * rec / max(1e-8, prec + rec)

    print(f"\n--- M1 Evaluation Results ---")
    print(f"Candidates: {len(m1_surviving)}")
    print(f"TP: {m1_tp} (Delta vs Baseline: {m1_tp - b_tp:+d})")
    print(f"FP: {m1_fp} (Delta vs Baseline: {m1_fp - b_fp:+d})")
    print(f"FN: {m1_fn}")
    print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f}")

    # Audit the 17 individual TPs
    b_keys = {f"{m['image']}:GT#{m['gt_idx']}(cls{m['class_id']})" for m in b_matches}
    m1_keys = {f"{m['image']}:GT#{m['gt_idx']}(cls{m['class_id']})" for m in m1_matches}
    lost_tps = b_keys - m1_keys
    gained_tps = m1_keys - b_keys
    print(f"\nTP Changes: Lost={len(lost_tps)}, Gained={len(gained_tps)}")
    if lost_tps:
        print(f"  Lost signs: {lost_tps}")
    else:
        print("  All 17 baseline TPs individually preserved with zero loss!")

    # Check Class 12 diamond signs & 00014 cls 21
    c12_signs = {"00008.ppm:GT#1(cls12)", "00010.ppm:GT#0(cls12)", "00016.ppm:GT#0(cls12)", "00019.ppm:GT#0(cls12)"}
    c12_retained = len(c12_signs & m1_keys)
    cls21_ok = "00014.ppm:GT#0(cls21)" in m1_keys
    print(f"Class 12 Retention: {c12_retained}/4")
    print(f"00014.ppm Class 21 Color Detection: {'RETAINED' if cls21_ok else 'LOST'}")

    # Save summary CSVs
    # 1. Summary comparison
    summary_csv = out_dir / 'phase_m1_comparison_summary.csv'
    with open(summary_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['configuration', 'candidates', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'delta_tp', 'delta_fp', 'class12_kept', 'cls21_retained'])
        w.writerow(['Baseline_Phase_L_C2', len(baseline_records), b_tp, b_fp, b_fn, f"{b_tp/len(baseline_records):.4f}", f"{b_tp/29:.4f}", f"{2*(b_tp/len(baseline_records))*(b_tp/29)/(b_tp/len(baseline_records)+b_tp/29):.4f}", 0, 0, '4/4', True])
        w.writerow(['Phase_M1_Corridor_0.25_0.70', len(m1_surviving), m1_tp, m1_fp, m1_fn, f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", m1_tp - b_tp, m1_fp - b_fp, f"{c12_retained}/4", cls21_ok])
    print(f"\nSaved comparison summary to: {summary_csv}")

    # 2. Per-image CSV
    per_img_csv = out_dir / 'phase_m1_per_image.csv'
    with open(per_img_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        w.writeheader()
        w.writerows(per_image_rows)
    print(f"Saved per-image metrics to: {per_img_csv}")

    # 3. Individual TP audit CSV
    tp_audit_csv = out_dir / 'phase_m1_tp_audit.csv'
    with open(tp_audit_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'gt_idx', 'class_id', 'source', 'iou', 'det_box', 'center_y_ratio', 'score_c2', 'baseline_detected', 'm1_detected'])
        for b_m in b_matches:
            key = f"{b_m['image']}:GT#{b_m['gt_idx']}(cls{b_m['class_id']})"
            is_in_m1 = key in m1_keys
            w.writerow([
                b_m['image'], b_m['gt_idx'], b_m['class_id'], b_m['source'],
                f"{b_m['iou']:.4f}", b_m['det_box'], f"{b_m['center_y_ratio']:.4f}",
                f"{b_m['score_c2']:.4f}", True, is_in_m1
            ])
    print(f"Saved individual TP audit to: {tp_audit_csv}")

    # 4. Rejections log CSV
    rej_csv = out_dir / 'phase_m1_rejections_log.csv'
    with open(rej_csv, 'w', newline='') as f:
        keys = ['image', 'x', 'y', 'w', 'h', 'source', 'center_y_ratio', 'score_c2', 'm1_rejection_reason']
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(m1_rejected)
    print(f"Saved rejections log to: {rej_csv}")


if __name__ == '__main__':
    run_phase_m1()
