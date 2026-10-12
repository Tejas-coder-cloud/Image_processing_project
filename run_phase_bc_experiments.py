"""Controlled experiment runner comparing:
1. Baseline (Combined mask, k=5, close=3, open=1, Filter 9 current)
2. Previous Best (Red-only reference with F2 alternative filter 9)
3. New Phase B & C Config: Independent Color Morphology (Red + Blue + Yellow),
   k=3, close=1, open=1, Filter 9 Alternative, Filter 11 max_area=8000, and NMS box deduplication (IoU=0.40).

Evaluates on the exact same 20 GTSDB images (29 GT signs) with strict one-to-one IoU matching at 0.5.
Saves overlays, per-image CSVs, and master summary in experiments_phase_bc/.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from src.config import PipelineConfig, DatasetConfig
from src.pipeline import run_full_pipeline, run_stage1
from src.stage2_segmentation.thresholding import (
    threshold_hsv_red, threshold_lab_red, threshold_hsv_white, _create_vegetation_mask
)
from src.stage2_segmentation.morphology import (
    apply_morphology, extract_candidate_regions, non_max_suppression_boxes
)
from evaluate_and_compare import load_gtsdb_annotations, compute_iou_xywh, draw_overlay


def evaluate_one_to_one(
    detected_bboxes: List[Tuple[int, int, int, int]],
    gt_bboxes: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Strict one-to-one greedy bipartite IoU matching."""
    n_gt = len(gt_bboxes)
    n_det = len(detected_bboxes)

    iou_pairs = []
    for gi, gt in enumerate(gt_bboxes):
        gt_box = (gt["x"], gt["y"], gt["w"], gt["h"])
        for di, det_box in enumerate(detected_bboxes):
            iou = compute_iou_xywh(gt_box, det_box)
            if iou >= iou_threshold:
                iou_pairs.append((iou, gi, di))

    iou_pairs.sort(key=lambda item: item[0], reverse=True)

    matched_gt = set()
    matched_det = set()
    matches = []

    for iou, gi, di in iou_pairs:
        if gi in matched_gt or di in matched_det:
            continue
        matched_gt.add(gi)
        matched_det.add(di)
        matches.append({
            "gt_idx": gi,
            "det_idx": di,
            "iou": iou,
            "class_id": gt_bboxes[gi]["class_id"],
        })

    tp = len(matches)
    fp = n_det - len(matched_det)
    fn = n_gt - len(matched_gt)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
        "unmatched_gt_indices": [i for i in range(n_gt) if i not in matched_gt],
        "unmatched_det_indices": [i for i in range(n_det) if i not in matched_det],
    }


def run_experiment(
    exp_name: str,
    pipeline_cfg: PipelineConfig,
    sample_images: List[str],
    all_gt: Dict[str, List[Dict]],
    gtsdb_dir: Path,
    out_dir: Path,
    is_red_only_ref: bool = False,
) -> Dict[str, Any]:
    """Execute a single experiment condition across all sample images."""
    cond_dir = out_dir / exp_name
    cond_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = cond_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    rows = []
    total_gt = 0
    total_det = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    print("=" * 80)
    print(f"RUNNING: {exp_name}")
    print("=" * 80)

    for idx, img_name in enumerate(sample_images, 1):
        img_path = gtsdb_dir / img_name
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        gt_boxes = all_gt.get(img_name, [])

        if is_red_only_ref:
            # Recreate exact previous best: Red-only, k=5, close=3, open=1, Filter 9 alternative
            s1 = run_stage1(img_bgr)
            enh = s1["enhanced_image"]
            hsv = cv2.cvtColor(enh, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(enh, cv2.COLOR_BGR2LAB)
            hr = threshold_hsv_red(hsv, pipeline_cfg.stage2.hsv_red_lower1, pipeline_cfg.stage2.hsv_red_upper1,
                                   pipeline_cfg.stage2.hsv_red_lower2, pipeline_cfg.stage2.hsv_red_upper2)
            lr = threshold_lab_red(lab, pipeline_cfg.stage2.lab_red_lower, pipeline_cfg.stage2.lab_red_upper)
            fr = cv2.bitwise_and(hr, lr)
            veg = _create_vegetation_mask(hsv)
            fr_sub = cv2.bitwise_and(fr, cv2.bitwise_not(veg))
            hw = threshold_hsv_white(hsv, pipeline_cfg.stage2.hsv_white_lower, pipeline_cfg.stage2.hsv_white_upper)
            cl_mask = apply_morphology(fr_sub, kernel_size=5, close_iters=3, open_iters=1)
            aux = {"mask_red": fr_sub, "mask_white": hw, "mask_combined": fr_sub}
            bboxes, _, _ = extract_candidate_regions(
                cl_mask, config=pipeline_cfg.stage2, aux_masks=aux,
                image_shape=img_bgr.shape[:2], enhanced_bgr=enh
            )
        else:
            # Use unified pipeline with configured Stage 1 & Stage 2 parameters
            res = run_full_pipeline(img_bgr, pipeline_cfg)
            enh = res["stage1"]["enhanced_image"]
            bboxes = res["stage2"]["bounding_boxes"]

        eval_res = evaluate_one_to_one(bboxes, gt_boxes, iou_threshold=0.5)

        total_gt += len(gt_boxes)
        total_det += len(bboxes)
        total_tp += eval_res["tp"]
        total_fp += eval_res["fp"]
        total_fn += eval_res["fn"]

        overlay = draw_overlay(enh, bboxes, gt_boxes, eval_res, title=f"{exp_name}: {img_name}")
        cv2.imwrite(str(overlays_dir / f"{Path(img_name).stem}_overlay.png"), overlay)

        rows.append({
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

        if eval_res["tp"] > 0:
            match_info = ", ".join([f"GT#{m['gt_idx']} cls={m['class_id']} (IoU={m['iou']:.2f})" for m in eval_res["matches"]])
            print(f"  [{idx:2d}/20] {img_name} -> DET={len(bboxes):3d} | TP={eval_res['tp']} | {match_info}")
        else:
            print(f"  [{idx:2d}/20] {img_name} -> DET={len(bboxes):3d} | TP=0")

    # Save per-image CSV
    with open(cond_dir / f"{exp_name}_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    prec = total_tp / max(1, total_tp + total_fp)
    rec = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)

    print(f"\n[{exp_name} Summary] GT={total_gt} Det={total_det} TP={total_tp} FP={total_fp} FN={total_fn} P={prec:.4f} R={rec:.4f} F1={f1:.4f}\n")

    return {
        "experiment": exp_name,
        "total_gt": total_gt,
        "total_det": total_det,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def main():
    root = Path(__file__).resolve().parent
    ds_cfg = DatasetConfig()
    gtsdb_dir = root / ds_cfg.gtsdb_dir
    gt_file = root / ds_cfg.gtsdb_gt_file
    out_dir = root / "experiments_phase_bc"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_images = [f"{i:05d}.ppm" for i in range(20)]
    all_gt, _ = load_gtsdb_annotations(gt_file, gtsdb_dir)

    results = []

    # 1. Baseline: Combined mask, k=5, close=3, open=1, original Filter 9
    cfg_base = PipelineConfig()
    cfg_base.stage2.independent_color_morphology = False
    cfg_base.stage2.morph_kernel_size = 5
    cfg_base.stage2.morph_close_iterations = 3
    cfg_base.stage2.morph_open_iterations = 1
    cfg_base.stage2.filter9_mode = "current"
    cfg_base.stage2.filter11_max_area = 5000
    res_base = run_experiment("1_Baseline_Combined_k5_c3", cfg_base, sample_images, all_gt, gtsdb_dir, out_dir)
    results.append(res_base)

    # 2. Previous Best: Red-only reference with Alternative Filter 9 (k=5, close=3, open=1)
    cfg_prev_best = PipelineConfig()
    cfg_prev_best.stage2.filter9_mode = "alternative"
    cfg_prev_best.stage2.filter11_max_area = 5000
    res_prev = run_experiment("2_Previous_Best_RedOnly_F2", cfg_prev_best, sample_images, all_gt, gtsdb_dir, out_dir, is_red_only_ref=True)
    results.append(res_prev)

    # 3. New Phase B & C Proposed Configuration:
    # - Independent color morphology across Red, Blue, and Yellow
    # - Smaller morphology kernel k=3, close=1, open=1 to prevent tree/shadow bridging
    # - Filter 9 Alternative (geometry-aware vehicle filter preserving compact mid-frame signs)
    # - Filter 11 max area expanded to 8000 to prevent killing large real signs
    # - NMS IoU threshold = 0.40 to deduplicate cross-channel bounding boxes
    cfg_prop = PipelineConfig()
    cfg_prop.stage2.independent_color_morphology = True
    cfg_prop.stage2.morph_kernel_size = 3
    cfg_prop.stage2.morph_close_iterations = 1
    cfg_prop.stage2.morph_open_iterations = 1
    cfg_prop.stage2.filter9_mode = "alternative"
    cfg_prop.stage2.filter11_max_area = 8000
    cfg_prop.stage2.nms_iou_threshold = 0.40
    res_prop = run_experiment("3_Proposed_Independent_k3_c1_F9Alt", cfg_prop, sample_images, all_gt, gtsdb_dir, out_dir)
    results.append(res_prop)

    # Master comparison CSV
    summary_path = out_dir / "phase_bc_comparison_summary.csv"
    with open(summary_path, "w", newline="") as f:
        fields = ["experiment", "total_gt", "total_det", "tp", "fp", "fn", "precision", "recall", "f1"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = dict(r)
            row["precision"] = f"{r['precision']:.4f}"
            row["recall"] = f"{r['recall']:.4f}"
            row["f1"] = f"{r['f1']:.4f}"
            w.writerow(row)

    print("=" * 95)
    print("PHASE B & C CONTROLLED EXPERIMENT SUITE SUMMARY (IoU >= 0.5)")
    print("=" * 95)
    print(f"{'Experiment':<35} | {'Det':<6} | {'TP':<4} | {'FP':<6} | {'FN':<4} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 95)
    for r in results:
        print(f"{r['experiment']:<35} | {r['total_det']:<6d} | {r['tp']:<4d} | {r['fp']:<6d} | {r['fn']:<4d} | {r['precision']:<10.4f} | {r['recall']:<10.4f} | {r['f1']:<10.4f}")
    print("=" * 95)
    print(f"Comparison report saved to: {summary_path}\n")


if __name__ == "__main__":
    main()
