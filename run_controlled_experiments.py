"""Controlled ablation experiment suite for GTSDB traffic sign detection.

Implements controlled one-factor-at-a-time experiments on the 20-image sample:
  Baseline: Default pipeline (with veg subtraction, morph k=5 close=3 open=1, combined mask)
  Exp A: Disable morphological opening only (open=0)
  Exp B: Disable morphological closing only (close=0)
  Exp C: Test smaller closing kernel (k=3, close=1)
  Exp D1: Red-only mask (isolate red chromaticity)
  Exp D2: Blue-only mask (isolate blue chromaticity)
  Exp D3: Yellow-only mask (isolate yellow chromaticity)
  Exp D4: Combined mask (explicit baseline reference)

Evaluates with verified GTSDB ground-truth and strict one-to-one IoU matching at 0.5.
Saves mask images, candidate overlays, and CSV metrics to separate results folders.
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.config import PipelineConfig, Stage2Config, DatasetConfig
from src.pipeline import run_stage1
from src.stage2_segmentation.thresholding import (
    threshold_hsv_red, threshold_hsv_blue, threshold_hsv_yellow, threshold_hsv_white,
    threshold_lab_red, threshold_lab_blue, threshold_lab_yellow, _create_vegetation_mask
)
from src.stage2_segmentation.morphology import apply_morphology, extract_candidate_regions, draw_segmentation_overlay
from evaluate_and_compare import load_gtsdb_annotations, evaluate_one_to_one, draw_overlay


def run_single_experiment(
    exp_name: str,
    exp_dir: Path,
    sample_images: List[str],
    all_gt: Dict[str, List[Dict]],
    gtsdb_dir: Path,
    color_mode: str = "combined",  # "combined", "red", "blue", "yellow"
    kernel_size: int = 5,
    close_iters: int = 3,
    open_iters: int = 1,
    subtract_vegetation: bool = True,
    iou_thresh: float = 0.5,
) -> Dict[str, Any]:
    """Execute a single controlled experiment on the sample images."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = exp_dir / "masks"
    overlays_dir = exp_dir / "overlays"
    masks_dir.mkdir(exist_ok=True)
    overlays_dir.mkdir(exist_ok=True)

    cfg = Stage2Config()
    cfg.morph_kernel_size = kernel_size
    cfg.morph_close_iterations = close_iters
    cfg.morph_open_iterations = open_iters
    cfg.subtract_vegetation = subtract_vegetation

    csv_rows = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_det = 0

    print(f"--> Running Experiment: {exp_name}")
    print(f"    Settings: color_mode={color_mode}, k={kernel_size}, close={close_iters}, open={open_iters}, veg_sub={subtract_vegetation}")

    for idx, img_name in enumerate(sample_images, 1):
        img_path = gtsdb_dir / img_name
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]
        s1 = run_stage1(img_bgr)
        enh = s1["enhanced_image"]
        hsv = cv2.cvtColor(enh, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(enh, cv2.COLOR_BGR2LAB)

        # Color masks
        hr = threshold_hsv_red(hsv, cfg.hsv_red_lower1, cfg.hsv_red_upper1, cfg.hsv_red_lower2, cfg.hsv_red_upper2)
        hb = threshold_hsv_blue(hsv, cfg.hsv_blue_lower, cfg.hsv_blue_upper)
        hy = threshold_hsv_yellow(hsv, cfg.hsv_yellow_lower, cfg.hsv_yellow_upper)
        hw = threshold_hsv_white(hsv, cfg.hsv_white_lower, cfg.hsv_white_upper)

        lr = threshold_lab_red(lab, cfg.lab_red_lower, cfg.lab_red_upper)
        lb = threshold_lab_blue(lab, cfg.lab_blue_lower, cfg.lab_blue_upper)
        ly = threshold_lab_yellow(lab, cfg.lab_yellow_lower, cfg.lab_yellow_upper)

        fr = cv2.bitwise_and(hr, lr)
        fb = cv2.bitwise_and(hb, lb)
        fy = cv2.bitwise_and(hy, ly)

        if subtract_vegetation:
            veg = _create_vegetation_mask(hsv)
            inv_veg = cv2.bitwise_not(veg)
            fr = cv2.bitwise_and(fr, inv_veg)
            fb = cv2.bitwise_and(fb, inv_veg)
            fy = cv2.bitwise_and(fy, inv_veg)

        # Select color mask according to mode
        if color_mode == "red":
            raw_mask = fr
        elif color_mode == "blue":
            raw_mask = fb
        elif color_mode == "yellow":
            raw_mask = fy
        else:  # combined
            raw_mask = cv2.bitwise_or(fr, cv2.bitwise_or(fb, fy))

        # Morphology
        cl_mask = apply_morphology(
            raw_mask,
            kernel_size=kernel_size,
            close_iters=close_iters,
            open_iters=open_iters,
        )

        aux = {
            "mask_red": fr,
            "mask_blue": fb,
            "mask_yellow": fy,
            "mask_white": hw,
            "mask_combined": raw_mask,
        }

        # Candidate extraction
        bboxes, contours, verified_mask = extract_candidate_regions(
            cl_mask, config=cfg, aux_masks=aux, image_shape=(h, w), enhanced_bgr=enh
        )

        # Evaluate against GT
        gt_boxes = all_gt.get(img_name, [])
        eval_res = evaluate_one_to_one(bboxes, gt_boxes, iou_threshold=iou_thresh)

        total_gt += len(gt_boxes)
        total_det += len(bboxes)
        total_tp += eval_res["tp"]
        total_fp += eval_res["fp"]
        total_fn += eval_res["fn"]

        # Draw overlay
        stem = Path(img_name).stem
        overlay = draw_overlay(enh, bboxes, gt_boxes, eval_res, title=exp_name)
        cv2.imwrite(str(overlays_dir / f"{stem}_overlay.png"), overlay)
        cv2.imwrite(str(masks_dir / f"{stem}_mask.png"), verified_mask)

        csv_rows.append({
            "image": img_name,
            "gt_signs": len(gt_boxes),
            "detected": len(bboxes),
            "tp": eval_res["tp"],
            "fp": eval_res["fp"],
            "fn": eval_res["fn"],
            "precision": f"{eval_res['precision']:.4f}",
            "recall": f"{eval_res['recall']:.4f}",
            "f1": f"{eval_res['f1']:.4f}",
        })

    # Save summary CSV
    csv_file = exp_dir / f"{exp_name}_summary.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    prec = total_tp / max(1, total_tp + total_fp)
    rec = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)

    result_summary = {
        "exp_name": exp_name,
        "color_mode": color_mode,
        "kernel_size": kernel_size,
        "close_iters": close_iters,
        "open_iters": open_iters,
        "total_gt": total_gt,
        "total_det": total_det,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "csv_path": str(csv_file),
    }

    print(f"    [Result] GT={total_gt} Det={total_det} TP={total_tp} FP={total_fp} FN={total_fn} P={prec:.4f} R={rec:.4f} F1={f1:.4f}\n")
    return result_summary


def main():
    root = Path(__file__).resolve().parent
    gtsdb_dir = root / "data" / "TrainIJCNN2013" / "TrainIJCNN2013"
    gt_file = gtsdb_dir / "gt.txt"
    base_results_dir = root / "experiments_suite"
    base_results_dir.mkdir(parents=True, exist_ok=True)

    sample_images = [f"{i:05d}.ppm" for i in range(20)]
    all_gt, val_stats = load_gtsdb_annotations(gt_file, gtsdb_dir)

    experiments_to_run = [
        # Baseline
        {
            "name": "Baseline",
            "color_mode": "combined",
            "kernel_size": 5,
            "close_iters": 3,
            "open_iters": 1,
            "veg_sub": True,
        },
        # Exp A: Disable morphological opening only (open=0)
        {
            "name": "Exp_A_No_Opening",
            "color_mode": "combined",
            "kernel_size": 5,
            "close_iters": 3,
            "open_iters": 0,
            "veg_sub": True,
        },
        # Exp B: Disable morphological closing only (close=0)
        {
            "name": "Exp_B_No_Closing",
            "color_mode": "combined",
            "kernel_size": 5,
            "close_iters": 0,
            "open_iters": 1,
            "veg_sub": True,
        },
        # Exp C: Test smaller closing kernel (k=3, close=1)
        {
            "name": "Exp_C_Small_Closing_k3",
            "color_mode": "combined",
            "kernel_size": 3,
            "close_iters": 1,
            "open_iters": 1,
            "veg_sub": True,
        },
        # Exp D1: Red-only mask
        {
            "name": "Exp_D1_Red_Only",
            "color_mode": "red",
            "kernel_size": 5,
            "close_iters": 3,
            "open_iters": 1,
            "veg_sub": True,
        },
        # Exp D2: Blue-only mask
        {
            "name": "Exp_D2_Blue_Only",
            "color_mode": "blue",
            "kernel_size": 5,
            "close_iters": 3,
            "open_iters": 1,
            "veg_sub": True,
        },
        # Exp D3: Yellow-only mask
        {
            "name": "Exp_D3_Yellow_Only",
            "color_mode": "yellow",
            "kernel_size": 5,
            "close_iters": 3,
            "open_iters": 1,
            "veg_sub": True,
        },
        # Exp D4: Combined mask (explicit baseline reference)
        {
            "name": "Exp_D4_Combined_Mask",
            "color_mode": "combined",
            "kernel_size": 5,
            "close_iters": 3,
            "open_iters": 1,
            "veg_sub": True,
        },
    ]

    all_exp_summaries = []

    for exp in experiments_to_run:
        exp_dir = base_results_dir / exp["name"]
        summary = run_single_experiment(
            exp_name=exp["name"],
            exp_dir=exp_dir,
            sample_images=sample_images,
            all_gt=all_gt,
            gtsdb_dir=gtsdb_dir,
            color_mode=exp["color_mode"],
            kernel_size=exp["kernel_size"],
            close_iters=exp["close_iters"],
            open_iters=exp["open_iters"],
            subtract_vegetation=exp["veg_sub"],
            iou_thresh=0.5,
        )
        all_exp_summaries.append(summary)

    # Master comparison CSV
    master_csv = base_results_dir / "all_experiments_comparison.csv"
    with open(master_csv, "w", newline="") as f:
        fields = ["exp_name", "color_mode", "kernel_size", "close_iters", "open_iters",
                  "total_gt", "total_det", "tp", "fp", "fn", "precision", "recall", "f1"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in all_exp_summaries:
            row = {k: s[k] for k in fields}
            row["precision"] = f"{s['precision']:.4f}"
            row["recall"] = f"{s['recall']:.4f}"
            row["f1"] = f"{s['f1']:.4f}"
            writer.writerow(row)

    print("\n" + "=" * 95)
    print("ALL EXPERIMENTS COMPARISON SUMMARY (IoU >= 0.5)")
    print("=" * 95)
    header = f"{'Experiment':<25} | {'Mode':<9} | {'Params':<16} | {'Det':<5} | {'TP':<3} | {'FP':<5} | {'FN':<3} | {'Prec':<7} | {'Rec':<7} | {'F1':<7}"
    print(header)
    print("-" * 95)
    for s in all_exp_summaries:
        param_str = f"k={s['kernel_size']},c={s['close_iters']},o={s['open_iters']}"
        print(f"{s['exp_name']:<25} | {s['color_mode']:<9} | {param_str:<16} | {s['total_det']:<5d} | {s['tp']:<3d} | {s['fp']:<5d} | {s['fn']:<3d} | {s['precision']:<7.4f} | {s['recall']:<7.4f} | {s['f1']:<7.4f}")
    print("=" * 95)
    print(f"Master CSV saved to: {master_csv}\n")


if __name__ == "__main__":
    main()
