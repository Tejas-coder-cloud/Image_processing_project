"""Audit runner to reproduce M3_D baseline without overwriting existing results.

This script replicates the core logic from run_phase_m3_experiment.py
but writes outputs to experiments_phase_n1/baseline_audit/ for safe verification.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from src.config import PipelineConfig, DatasetConfig
from evaluate_and_compare import load_gtsdb_annotations, compute_iou_xywh, draw_overlay
from src.pipeline import run_stage1
from src.stage2_segmentation.thresholding import segment_traffic_sign_colors
from run_phase_l_expc import (
    process_all_images, evaluate_one_to_one
)
from run_phase_m2_experiment import compute_color_mask_solidity


def run_m3d_audit():
    root = Path('.').resolve()
    ds_cfg = DatasetConfig()
    gtsdb_dir = root / ds_cfg.gtsdb_dir
    gt_file = root / ds_cfg.gtsdb_gt_file
    out_dir = root / 'experiments_phase_n1' / 'baseline_audit'
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / 'm3d_overlays'
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

    print("M3_D Baseline Audit: Extracting base candidates and computing features...")
    all_records = process_all_images(cfg, sample_images, all_gt, gtsdb_dir)

    # 1. Apply Phase L C2 score >= 0.200
    c2_survivors = [r for r in all_records if float(r['score_c2']) >= 0.200]

    # 2. Apply Phase M1 Vertical Corridor [0.25, 0.70]
    m1_survivors = [r for r in c2_survivors if 0.25 <= float(r['center_y_ratio']) <= 0.70]

    # 3. Apply Phase M2_D Color Rules
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
            x, y, w, h = int(r['x']), int(r['y']), int(r['w']), int(r['h'])
            crop_m = mask_comb[y:y+h, x:x+w]
            sol = compute_color_mask_solidity(crop_m)
            sym = float(r['edge_symmetry'])
            ar_dev = float(r['ar_deviation'])
            # M2_D rule
            if sol >= 0.20 and sym >= 0.12 and ar_dev <= 0.42:
                m2_d_survivors.append(r)
        else:
            # Edge candidates pass M2_D untouched
            m2_d_survivors.append(r)

    print(f"M2_D Baseline candidates: {len(m2_d_survivors)}")
    print(f"Expected: 1457")
    if len(m2_d_survivors) != 1457:
        print(f"WARNING: M2_D count mismatch! Got {len(m2_d_survivors)}, expected 1457")

    # 4. Apply Phase M3_D Edge Filters
    m3_d_survivors = [
        r for r in m2_d_survivors
        if r['source'] == 'color' or (
            float(r['edge_symmetry']) >= 0.18 and
            float(r['ar_deviation']) <= 0.48 and
            float(r['box_area']) <= 15000.0
        )
    ]

    print(f"M3_D candidates: {len(m3_d_survivors)}")
    print(f"Expected: 1355")
    if len(m3_d_survivors) != 1355:
        print(f"WARNING: M3_D count mismatch! Got {len(m3_d_survivors)}, expected 1355")

    # Evaluate M3_D
    by_img: Dict[str, List[Dict]] = {}
    for r in m3_d_survivors:
        by_img.setdefault(r['image'], []).append(r)

    c_tp = c_fp = c_fn = 0
    matches_keys = set()
    per_image_summary = []

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
            'cands': len(recs), 'tp': ev['tp'], 'fp': ev['fp'], 'fn': ev['fn'],
            'precision': f"{ev['precision']:.4f}", 'recall': f"{ev['recall']:.4f}"
        })

        # Save overlay
        img_bgr = cv2.imread(str(gtsdb_dir / img_name))
        ov = draw_overlay(img_bgr, det_boxes, gt_boxes, ev, title=f"M3_D Audit: {img_name}")
        cv2.imwrite(str(overlay_dir / f"{Path(img_name).stem}_m3d_audit_overlay.png"), ov)

    assert c_tp + c_fn == 29, f"TP+FN != 29 for M3_D audit"
    assert c_tp + c_fp == len(m3_d_survivors), f"TP+FP != surviving for M3_D audit"

    prec = c_tp / max(1, c_tp + c_fp)
    rec = c_tp / 29.0
    f1 = 2 * prec * rec / max(1e-8, prec + rec)

    # Audit Class 12 and Class 21
    c12_signs = {"00008.ppm:GT#1(cls12)", "00010.ppm:GT#0(cls12)", "00016.ppm:GT#0(cls12)", "00019.ppm:GT#0(cls12)"}
    cls21_ref = "00014.ppm:GT#0(cls21)"
    c12_count = len(c12_signs & matches_keys)
    cls21_ok = cls21_ref in matches_keys

    print("\n" + "="*80)
    print("M3_D BASELINE AUDIT RESULTS")
    print("="*80)
    print(f"Candidates: {len(m3_d_survivors)} (expected: 1355)")
    print(f"TP: {c_tp} (expected: 17)")
    print(f"FP: {c_fp} (expected: 1338)")
    print(f"FN: {c_fn} (expected: 12)")
    print(f"Precision: {prec:.4f} (expected: ~0.0125)")
    print(f"Recall: {rec:.4f} (expected: 0.5862)")
    print(f"F1: {f1:.4f} (expected: ~0.0246)")
    print(f"Class 12 retained: {c12_count}/4 (expected: 4/4)")
    print(f"Class 21 (00014.ppm): {'OK' if cls21_ok else 'LOST'} (expected: OK)")
    print("="*80)

    # Save results
    summary_csv = out_dir / 'm3d_audit_summary.csv'
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['metric', 'value', 'expected'])
        w.writeheader()
        w.writerows([
            {'metric': 'candidates', 'value': len(m3_d_survivors), 'expected': 1355},
            {'metric': 'tp', 'value': c_tp, 'expected': 17},
            {'metric': 'fp', 'value': c_fp, 'expected': 1338},
            {'metric': 'fn', 'value': c_fn, 'expected': 12},
            {'metric': 'precision', 'value': f"{prec:.4f}", 'expected': '0.0125'},
            {'metric': 'recall', 'value': f"{rec:.4f}", 'expected': '0.5862'},
            {'metric': 'f1', 'value': f"{f1:.4f}", 'expected': '0.0246'},
            {'metric': 'class12_retained', 'value': f"{c12_count}/4", 'expected': '4/4'},
            {'metric': 'cls21_00014', 'value': 'OK' if cls21_ok else 'LOST', 'expected': 'OK'},
        ])
    print(f"\nAudit summary saved: {summary_csv}")

    per_img_csv = out_dir / 'm3d_audit_per_image.csv'
    with open(per_img_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_summary[0].keys()))
        w.writeheader()
        w.writerows(per_image_summary)
    print(f"Per-image metrics saved: {per_img_csv}")

    # TP audit
    tp_audit_csv = out_dir / 'm3d_audit_tp_audit.csv'
    with open(tp_audit_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_gt', 'detected'])
        for key in sorted(matches_keys):
            w.writerow([key, True])
    print(f"TP audit saved: {tp_audit_csv}")

    # Check for discrepancies
    discrepancies = []
    if len(m3_d_survivors) != 1355:
        discrepancies.append(f"Candidate count: {len(m3_d_survivors)} vs 1355")
    if c_tp != 17:
        discrepancies.append(f"TP: {c_tp} vs 17")
    if c_fp != 1338:
        discrepancies.append(f"FP: {c_fp} vs 1338")
    if c_fn != 12:
        discrepancies.append(f"FN: {c_fn} vs 12")
    if c12_count != 4:
        discrepancies.append(f"Class 12: {c12_count}/4 vs 4/4")
    if not cls21_ok:
        discrepancies.append("Class 21 (00014.ppm): LOST vs OK")

    if discrepancies:
        print("\n⚠️  DISCREPANCIES DETECTED:")
        for d in discrepancies:
            print(f"  - {d}")
    else:
        print("\n✓ Baseline reproduced successfully!")

    return len(m3_d_survivors), c_tp, c_fp, c_fn, prec, rec, f1, c12_count, cls21_ok


if __name__ == '__main__':
    run_m3d_audit()
