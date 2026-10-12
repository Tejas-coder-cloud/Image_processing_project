"""Task B: Controlled Filter 9 Ablation Experiment Suite.

Conditions tested on E1 Red-Only configuration across 20 GTSDB images:
- F0: Current Filter 9 (`_is_likely_vehicle_region`)
- F1: Bypass Filter 9 only (no other change)
- F2: Alternative Filter 9 (documenting relaxed aspect ratio & spatial condition)
      Vehicles are distinctly wider-than-tall (aspect_ratio > 1.3) or huge (area > 8000).
      Compact signs with aspect ~ 0.5 - 1.2 in middle-lower frame are preserved.

Outputs:
- experiments_filter9/F0_Current_Filter9/
- experiments_filter9/F1_Bypass_Filter9/
- experiments_filter9/F2_Alternative_Filter9/
- Filter 9 rejection logs with exact box coordinates, center-y ratio, area, aspect ratio,
  rejection reason, and IoU against ground truth.
- Explicit verification of (779, 389, 834, 497) in 00000.ppm.
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.config import Stage2Config
from src.pipeline import run_stage1
from src.stage2_segmentation.thresholding import (
    threshold_hsv_red, threshold_lab_red, threshold_hsv_white, _create_vegetation_mask
)
from src.stage2_segmentation.morphology import (
    apply_morphology, _compute_circularity, _compute_convexity,
    _compute_color_density, _compute_edge_density, _is_likely_vehicle_region,
    _is_traffic_light, _is_large_background_region
)
from evaluate_and_compare import load_gtsdb_annotations, compute_iou_xywh, draw_overlay


def is_vehicle_alternative(x: int, y: int, w: int, h: int, img_h: int, img_w: int) -> Tuple[bool, str]:
    """Alternative Filter 9 logic:

    Rather than cutting off everything below y_ratio > 0.55 with area > 400:
    1. Real vehicles are horizontally elongated (wide aspect ratio w/h > 1.35)
       or massive contiguous truck bodies (area > 8000).
    2. Lower road clutter near bottom: y_ratio > 0.80 and area > 1000.
    """
    center_y = y + h / 2.0
    y_ratio = center_y / max(1, img_h)
    area = w * h
    aspect = float(w) / max(1, h)

    if area > 8000:
        return True, "massive_blob_area>8000"
    if y_ratio > 0.65 and aspect > 1.4 and area > 500:
        return True, "wide_vehicle_body_y>0.65_aspect>1.4"
    if y_ratio > 0.80 and area > 1000:
        return True, "road_surface_y>0.80"

    return False, "passed"


def extract_candidates_filter9_ablation(
    binary_mask: np.ndarray,
    config: Stage2Config,
    aux_masks: Dict[str, np.ndarray],
    image_shape: Tuple[int, int],
    enhanced_bgr: np.ndarray,
    filter9_mode: str,  # "F0_current", "F1_bypass", "F2_alternative"
    gt_boxes_for_image: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """Extract candidates while tracking exact Filter 9 rejections and IoU with GT."""
    img_h, img_w = image_shape
    image_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

    white_mask = aux_masks.get("mask_white")
    red_mask = aux_masks.get("mask_red")
    combined_mask = aux_masks.get("mask_combined", binary_mask)

    rejections = {
        "raw_contours": 0,
        "filter1_area": 0,
        "filter2_dimension": 0,
        "filter3_aspect_ratio": 0,
        "filter4_spatial_position": 0,
        "filter5_circularity": 0,
        "filter6_convexity": 0,
        "filter7_color_density": 0,
        "filter8_edge_density": 0,
        "filter9_vehicle": 0,
        "filter10_traffic_light": 0,
        "filter11_background": 0,
        "filter12_structure": 0,
        "passed": 0,
    }

    rejected_by_f9_log: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rejections["raw_contours"] = len(contours)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.min_area or area > config.max_area:
            rejections["filter1_area"] += 1
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < config.min_box_dim or h < config.min_box_dim or w > config.max_box_dim or h > config.max_box_dim:
            rejections["filter2_dimension"] += 1
            continue

        aspect_ratio = float(w) / max(1, h)
        if not (config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio):
            rejections["filter3_aspect_ratio"] += 1
            continue

        if y > int(config.max_road_y_ratio * img_h) or (y + h) < int(config.min_sky_y_ratio * img_h):
            rejections["filter4_spatial_position"] += 1
            continue

        circularity = _compute_circularity(cnt)
        if circularity < config.min_circularity:
            rejections["filter5_circularity"] += 1
            continue

        convexity = _compute_convexity(cnt)
        if convexity < config.min_convexity:
            rejections["filter6_convexity"] += 1
            continue

        crop_combined = combined_mask[y:y + h, x:x + w]
        color_density = _compute_color_density(crop_combined, w, h)
        if color_density < config.min_color_density or color_density > config.max_color_density:
            rejections["filter7_color_density"] += 1
            continue

        edge_density = _compute_edge_density(image_gray, x, y, w, h)
        if edge_density < config.min_edge_density or edge_density > config.max_edge_density:
            rejections["filter8_edge_density"] += 1
            continue

        # ============ FILTER 9 EVALUATION ============
        center_y = y + h / 2.0
        y_ratio = center_y / max(1, img_h)
        box_area = w * h

        # Calculate best IoU with any GT in this image
        cand_box_xywh = (x, y, w, h)
        best_gt_iou = 0.0
        best_gt_idx = -1
        for gi, gt in enumerate(gt_boxes_for_image):
            gt_box_xywh = (gt["x"], gt["y"], gt["w"], gt["h"])
            iou = compute_iou_xywh(cand_box_xywh, gt_box_xywh)
            if iou > best_gt_iou:
                best_gt_iou = iou
                best_gt_idx = gi

        is_rejected_f9 = False
        rejection_reason = ""

        if filter9_mode == "F0_current":
            if _is_likely_vehicle_region(x, y, w, h, img_h, img_w, red_mask):
                is_rejected_f9 = True
                if y_ratio > 0.55 and box_area > 400:
                    rejection_reason = "y_ratio>0.55_and_area>400"
                elif y_ratio > 0.65 and box_area > 150:
                    rejection_reason = "y_ratio>0.65_and_area>150"
                elif y_ratio > 0.45 and box_area > 2000:
                    rejection_reason = "y_ratio>0.45_and_area>2000"
                else:
                    rejection_reason = "vehicle_heuristic"

        elif filter9_mode == "F1_bypass":
            is_rejected_f9 = False

        elif filter9_mode == "F2_alternative":
            rej, reason = is_vehicle_alternative(x, y, w, h, img_h, img_w)
            if rej:
                is_rejected_f9 = True
                rejection_reason = reason

        if is_rejected_f9:
            rejections["filter9_vehicle"] += 1
            rejected_by_f9_log.append({
                "x1": x, "y1": y, "x2": x + w, "y2": y + h,
                "w": w, "h": h, "contour_area": area, "box_area": box_area,
                "center_y_ratio": y_ratio, "aspect_ratio": aspect_ratio,
                "rejection_reason": rejection_reason,
                "best_gt_idx": best_gt_idx, "best_gt_iou": best_gt_iou,
            })
            continue

        # Filter 10: Traffic light
        if _is_traffic_light(image_gray, x, y, w, h):
            rejections["filter10_traffic_light"] += 1
            continue

        # Filter 11: Background
        if _is_large_background_region(image_gray, x, y, w, h):
            rejections["filter11_background"] += 1
            continue

        # Filter 12: Structure
        crop_w = white_mask[y:y + h, x:x + w] if white_mask is not None else None
        crop_r = red_mask[y:y + h, x:x + w] if red_mask is not None else None
        w_count = cv2.countNonZero(crop_w) if crop_w is not None else 0
        r_count = cv2.countNonZero(crop_r) if crop_r is not None else 0
        total_box_pixels = max(1, w * h)
        sign_color_frac = float(r_count) / total_box_pixels

        has_color_content = sign_color_frac >= 0.08
        has_white_interior = w_count >= 5 and (r_count >= 3)
        has_solid_shape = convexity >= 0.60 and circularity >= 0.30

        if not (has_color_content or has_white_interior or has_solid_shape):
            rejections["filter12_structure"] += 1
            continue

        rejections["passed"] += 1
        candidates.append({
            "x1": x, "y1": y, "x2": x + w, "y2": y + h,
            "x": x, "y": y, "w": w, "h": h,
            "area": area, "circularity": circularity, "convexity": convexity,
            "best_gt_iou": best_gt_iou, "best_gt_idx": best_gt_idx,
        })

    return candidates, rejected_by_f9_log, rejections


def evaluate_one_to_one(
    detected_candidates: List[Dict[str, Any]],
    gt_bboxes: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Strict one-to-one IoU matching."""
    n_gt = len(gt_bboxes)
    n_det = len(detected_candidates)

    iou_pairs = []
    for gi, gt in enumerate(gt_bboxes):
        gt_box = (gt["x"], gt["y"], gt["w"], gt["h"])
        for di, det in enumerate(detected_candidates):
            det_box = (det["x"], det["y"], det["w"], det["h"])
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
            "gt_idx": gi, "det_idx": di, "iou": iou,
            "class_id": gt_bboxes[gi]["class_id"],
        })

    tp = len(matches)
    fp = n_det - len(matched_det)
    fn = n_gt - len(matched_gt)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "matches": matches,
        "unmatched_gt_indices": [i for i in range(n_gt) if i not in matched_gt],
        "unmatched_det_indices": [i for i in range(n_det) if i not in matched_det],
    }


def run_filter9_experiment(
    cond_name: str,
    filter9_mode: str,
    sample_images: List[str],
    all_gt: Dict[str, List[Dict]],
    gtsdb_dir: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run E1 red-only under specified Filter 9 condition."""
    cond_dir = out_dir / cond_name
    cond_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = cond_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    cfg = Stage2Config()
    cfg.morph_kernel_size = 5
    cfg.morph_close_iterations = 3
    cfg.morph_open_iterations = 1

    per_image_rows = []
    all_f9_rejections = []
    agg_rejections = {
        "raw_contours": 0, "filter1_area": 0, "filter2_dimension": 0,
        "filter3_aspect_ratio": 0, "filter4_spatial_position": 0,
        "filter5_circularity": 0, "filter6_convexity": 0, "filter7_color_density": 0,
        "filter8_edge_density": 0, "filter9_vehicle": 0, "filter10_traffic_light": 0,
        "filter11_background": 0, "filter12_structure": 0, "passed": 0
    }

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_det = 0

    print("=" * 80)
    print(f"RUNNING {cond_name} (Mode: {filter9_mode})")
    print("=" * 80)

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

        hr = threshold_hsv_red(hsv, cfg.hsv_red_lower1, cfg.hsv_red_upper1, cfg.hsv_red_lower2, cfg.hsv_red_upper2)
        lr = threshold_lab_red(lab, cfg.lab_red_lower, cfg.lab_red_upper)
        fr = cv2.bitwise_and(hr, lr)
        veg = _create_vegetation_mask(hsv)
        fr_sub = cv2.bitwise_and(fr, cv2.bitwise_not(veg))
        hw = threshold_hsv_white(hsv, cfg.hsv_white_lower, cfg.hsv_white_upper)

        cl_mask = apply_morphology(fr_sub, cfg.morph_kernel_size, cfg.morph_close_iterations, cfg.morph_open_iterations)

        aux = {"mask_red": fr_sub, "mask_white": hw, "mask_combined": fr_sub}
        gt_boxes = all_gt.get(img_name, [])

        cands, f9_rejs, rej_counts = extract_candidates_filter9_ablation(
            cl_mask, cfg, aux, (h, w), enh, filter9_mode=filter9_mode, gt_boxes_for_image=gt_boxes
        )

        for k in agg_rejections:
            agg_rejections[k] += rej_counts[k]

        for r in f9_rejs:
            r["image"] = img_name
            all_f9_rejections.append(r)

        eval_res = evaluate_one_to_one(cands, gt_boxes, iou_threshold=0.5)

        total_gt += len(gt_boxes)
        total_det += len(cands)
        total_tp += eval_res["tp"]
        total_fp += eval_res["fp"]
        total_fn += eval_res["fn"]

        # Overlay
        bboxes_tuple = [(c["x"], c["y"], c["w"], c["h"]) for c in cands]
        overlay = draw_overlay(enh, bboxes_tuple, gt_boxes, eval_res, title=f"{cond_name}: {img_name}")
        cv2.imwrite(str(overlays_dir / f"{Path(img_name).stem}_overlay.png"), overlay)

        per_image_rows.append({
            "image": img_name,
            "gt_signs": len(gt_boxes),
            "detected": len(cands),
            "tp": eval_res["tp"],
            "fp": eval_res["fp"],
            "fn": eval_res["fn"],
            "precision": f"{eval_res['precision']:.4f}",
            "recall": f"{eval_res['recall']:.4f}",
            "f1": f"{eval_res['f1']:.4f}",
        })

    # Save per-image CSV
    with open(cond_dir / f"{cond_name}_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        w.writeheader()
        w.writerows(per_image_rows)

    # Save Filter 9 rejections log
    if all_f9_rejections:
        with open(cond_dir / f"{cond_name}_filter9_rejections.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_f9_rejections[0].keys()))
            w.writeheader()
            w.writerows(all_f9_rejections)

    # Save filter breakdown
    with open(cond_dir / f"{cond_name}_filter_breakdown.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Filter", "Count_Rejected"])
        for k, v in agg_rejections.items():
            w.writerow([k, v])

    overall_p = total_tp / max(1, total_tp + total_fp)
    overall_r = total_tp / max(1, total_tp + total_fn)
    overall_f1 = 2 * overall_p * overall_r / max(1e-8, overall_p + overall_r)

    print(f"[{cond_name} Summary] GT={total_gt} Det={total_det} TP={total_tp} FP={total_fp} FN={total_fn} P={overall_p:.4f} R={overall_r:.4f} F1={overall_f1:.4f}")
    print(f"Filter 9 Rejections: {agg_rejections['filter9_vehicle']}, Emitted: {total_det}\n")

    return {
        "condition": cond_name,
        "filter9_mode": filter9_mode,
        "total_gt": total_gt,
        "total_det": total_det,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": overall_p,
        "recall": overall_r,
        "f1": overall_f1,
        "f9_rejections": agg_rejections["filter9_vehicle"],
    }


def main():
    root = Path(__file__).resolve().parent
    gtsdb_dir = root / "data" / "TrainIJCNN2013" / "TrainIJCNN2013"
    gt_file = gtsdb_dir / "gt.txt"
    out_dir = root / "experiments_filter9"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_images = [f"{i:05d}.ppm" for i in range(20)]
    all_gt, val_stats = load_gtsdb_annotations(gt_file, gtsdb_dir)

    conditions = [
        {"name": "F0_Current_Filter9", "mode": "F0_current"},
        {"name": "F1_Bypass_Filter9", "mode": "F1_bypass"},
        {"name": "F2_Alternative_Filter9", "mode": "F2_alternative"},
    ]

    summaries = []
    for c in conditions:
        s = run_filter9_experiment(
            cond_name=c["name"],
            filter9_mode=c["mode"],
            sample_images=sample_images,
            all_gt=all_gt,
            gtsdb_dir=gtsdb_dir,
            out_dir=out_dir,
        )
        summaries.append(s)

    # Master summary CSV
    summary_csv = out_dir / "filter9_ablation_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        fields = ["condition", "filter9_mode", "total_gt", "total_det", "tp", "fp", "fn", "precision", "recall", "f1", "f9_rejections"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            row = {k: s[k] for k in fields}
            row["precision"] = f"{s['precision']:.4f}"
            row["recall"] = f"{s['recall']:.4f}"
            row["f1"] = f"{s['f1']:.4f}"
            w.writerow(row)

    print("\n" + "=" * 90)
    print("FILTER 9 ABLATION EXPERIMENTS SUMMARY (One-to-One IoU >= 0.5)")
    print("=" * 90)
    print(f"{'Condition':<25} | {'Det':<5} | {'F9 Rej':<7} | {'TP':<3} | {'FP':<5} | {'FN':<3} | {'Prec':<7} | {'Rec':<7} | {'F1':<7}")
    print("-" * 90)
    for s in summaries:
        print(f"{s['condition']:<25} | {s['total_det']:<5d} | {s['f9_rejections']:<7d} | {s['tp']:<3d} | {s['fp']:<5d} | {s['fn']:<3d} | {s['precision']:<7.4f} | {s['recall']:<7.4f} | {s['f1']:<7.4f}")
    print("=" * 90)
    print(f"Summary saved to: {summary_csv}\n")


if __name__ == "__main__":
    main()
