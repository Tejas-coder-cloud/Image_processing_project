"""Controlled Color-Specific Morphology Experiments (E1, E2, E3, E4) on GTSDB.

Implements:
- E1: Red-only reference (k=5, close=3, open=1)
- E2: Red and Blue masks processed independently (k=5, close=3, open=1), candidate boxes combined (not pixel masks)
- E3: Red, Blue, and Yellow masks processed independently (k=5, close=3, open=1), candidate boxes combined
- E4: Red, Blue, and Yellow masks processed independently with 1 closing iteration and 3x3 kernel (k=3, close=1, open=1), candidate boxes combined

Features:
- Candidate provenance tracking (color_channel, morphology settings, image coordinates)
- Consistent (x1, y1, x2, y2) original-image coordinates
- Strict one-to-one IoU matching at IoU >= 0.5
- Filter rejection breakdown tracking
- Outputs saved in experiments_color_morphology/ without modifying original project or datasets
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.config import Stage2Config, DatasetConfig
from src.pipeline import run_stage1
from src.stage2_segmentation.thresholding import (
    threshold_hsv_red, threshold_hsv_blue, threshold_hsv_yellow, threshold_hsv_white,
    threshold_lab_red, threshold_lab_blue, threshold_lab_yellow, _create_vegetation_mask
)
from src.stage2_segmentation.morphology import (
    apply_morphology, _compute_circularity, _compute_convexity,
    _compute_color_density, _compute_edge_density, _is_likely_vehicle_region,
    _is_traffic_light, _is_large_background_region
)
from evaluate_and_compare import load_gtsdb_annotations, compute_iou_xywh


def extract_candidates_with_provenance_and_filter_tracking(
    binary_mask: np.ndarray,
    config: Stage2Config,
    aux_masks: Dict[str, np.ndarray],
    image_shape: Tuple[int, int],
    enhanced_bgr: np.ndarray,
    color_channel: str,
    morph_setting: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Extract candidate bounding boxes with full provenance and detailed filter rejection stats.

    Coordinates returned: (x1, y1, x2, y2) and (x, y, w, h).
    """
    img_h, img_w = image_shape
    image_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

    white_mask = aux_masks.get("mask_white")
    yellow_mask = aux_masks.get("mask_yellow")
    red_mask = aux_masks.get("mask_red")
    blue_mask = aux_masks.get("mask_blue")
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

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rejections["raw_contours"] = len(contours)

    is_cropped_patch = (img_h < 150 and img_w < 150)
    candidates = []

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

        if not is_cropped_patch:
            if y > int(config.max_road_y_ratio * img_h):
                rejections["filter4_spatial_position"] += 1
                continue
            if (y + h) < int(config.min_sky_y_ratio * img_h):
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

        if not is_cropped_patch and _is_likely_vehicle_region(x, y, w, h, img_h, img_w, red_mask):
            rejections["filter9_vehicle"] += 1
            continue

        if not is_cropped_patch and _is_traffic_light(image_gray, x, y, w, h):
            rejections["filter10_traffic_light"] += 1
            continue

        if not is_cropped_patch and _is_large_background_region(image_gray, x, y, w, h):
            rejections["filter11_background"] += 1
            continue

        if config.require_sign_structure and not is_cropped_patch:
            crop_w = white_mask[y:y + h, x:x + w] if white_mask is not None else None
            crop_y = yellow_mask[y:y + h, x:x + w] if yellow_mask is not None else None
            crop_r = red_mask[y:y + h, x:x + w] if red_mask is not None else None
            crop_b = blue_mask[y:y + h, x:x + w] if blue_mask is not None else None

            w_count = cv2.countNonZero(crop_w) if crop_w is not None else 0
            y_count = cv2.countNonZero(crop_y) if crop_y is not None else 0
            r_count = cv2.countNonZero(crop_r) if crop_r is not None else 0
            b_count = cv2.countNonZero(crop_b) if crop_b is not None else 0

            total_box_pixels = max(1, w * h)
            sign_color_frac = float(r_count + b_count + y_count) / total_box_pixels

            has_color_content = sign_color_frac >= 0.08
            has_white_interior = w_count >= 5 and (r_count >= 3 or b_count >= 3)
            has_solid_shape = convexity >= 0.60 and circularity >= 0.30
            has_yellow_content = y_count >= 5

            if not (has_color_content or has_white_interior or has_solid_shape or has_yellow_content):
                rejections["filter12_structure"] += 1
                continue

        rejections["passed"] += 1
        candidates.append({
            "x1": x,
            "y1": y,
            "x2": x + w,
            "y2": y + h,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "circularity": circularity,
            "convexity": convexity,
            "color_channel": color_channel,
            "morph_setting": morph_setting,
        })

    return candidates, rejections


def evaluate_one_to_one_provenance(
    detected_candidates: List[Dict[str, Any]],
    gt_bboxes: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Strict one-to-one IoU matching ensuring at most 1 GT per prediction and 1 prediction per GT."""
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
            "gt_idx": gi,
            "det_idx": di,
            "iou": iou,
            "class_id": gt_bboxes[gi]["class_id"],
            "det_candidate": detected_candidates[di],
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


def draw_annotated_overlay(
    image_bgr: np.ndarray,
    candidates: List[Dict[str, Any]],
    gt_bboxes: List[Dict[str, Any]],
    eval_res: Dict[str, Any],
    title: str = "",
) -> np.ndarray:
    """Render visualization with color-coded boxes and provenance tags."""
    overlay = image_bgr.copy()
    h, w = overlay.shape[:2]

    color_bgr_map = {
        "red": (0, 0, 255),
        "blue": (255, 120, 0),
        "yellow": (0, 220, 255),
        "combined": (200, 200, 200),
    }

    # TP in green
    for match in eval_res["matches"]:
        cand = match["det_candidate"]
        x, y, bw, bh = cand["x"], cand["y"], cand["w"], cand["h"]
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        txt = f"TP:{cand['color_channel']} (IoU:{match['iou']:.2f})"
        cv2.putText(overlay, txt, (x, max(14, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

    # FP in red or color tag
    for di in eval_res["unmatched_det_indices"]:
        cand = candidates[di]
        x, y, bw, bh = cand["x"], cand["y"], cand["w"], cand["h"]
        c_tag = color_bgr_map.get(cand["color_channel"], (0, 0, 255))
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 0, 220), 1)
        txt = f"FP:{cand['color_channel'][:1].upper()}"
        cv2.putText(overlay, txt, (x, max(12, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 220), 1, cv2.LINE_AA)

    # FN in cyan dashed
    for gi in eval_res["unmatched_gt_indices"]:
        gt = gt_bboxes[gi]
        gx, gy, gbw, gbh = gt["x"], gt["y"], gt["w"], gt["h"]
        for i in range(0, gbw, 8):
            cv2.line(overlay, (gx + i, gy), (min(gx + i + 4, gx + gbw), gy), (255, 255, 0), 2)
            cv2.line(overlay, (gx + i, gy + gbh), (min(gx + i + 4, gx + gbw), gy + gbh), (255, 255, 0), 2)
        for i in range(0, gbh, 8):
            cv2.line(overlay, (gx, gy + i), (gx, min(gy + i + 4, gy + gbh)), (255, 255, 0), 2)
            cv2.line(overlay, (gx + gbw, gy + i), (gx + gbw, min(gy + i + 4, gy + gbh)), (255, 255, 0), 2)
        txt = f"MISS:cls{gt['class_id']}"
        cv2.putText(overlay, txt, (gx, max(14, gy - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)

    # Banner
    banner = f"{title} | TP={eval_res['tp']} FP={eval_res['fp']} FN={eval_res['fn']} P={eval_res['precision']:.2f} R={eval_res['recall']:.2f}"
    cv2.rectangle(overlay, (0, h - 28), (w, h), (15, 23, 42), -1)
    cv2.putText(overlay, banner, (12, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (241, 245, 249), 1, cv2.LINE_AA)

    return overlay


def run_experiment(
    exp_key: str,
    exp_desc: str,
    color_channels: List[str],  # e.g. ["red"], ["red", "blue"], ["red", "blue", "yellow"]
    kernel_size: int,
    close_iters: int,
    open_iters: int,
    sample_images: List[str],
    all_gt: Dict[str, List[Dict]],
    gtsdb_dir: Path,
    out_dir: Path,
    iou_thresh: float = 0.5,
) -> Dict[str, Any]:
    """Execute one controlled experiment, logging per-color stats and rejection counts."""
    exp_dir = out_dir / exp_key
    exp_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = exp_dir / "masks"
    overlays_dir = exp_dir / "overlays"
    masks_dir.mkdir(exist_ok=True)
    overlays_dir.mkdir(exist_ok=True)

    cfg = Stage2Config()
    cfg.morph_kernel_size = kernel_size
    cfg.morph_close_iterations = close_iters
    cfg.morph_open_iterations = open_iters

    morph_tag = f"k={kernel_size}_c={close_iters}_o={open_iters}"

    per_image_rows = []
    candidates_provenance_rows = []
    aggregate_rejections = {
        "raw_contours": 0, "filter1_area": 0, "filter2_dimension": 0,
        "filter3_aspect_ratio": 0, "filter4_spatial_position": 0,
        "filter5_circularity": 0, "filter6_convexity": 0, "filter7_color_density": 0,
        "filter8_edge_density": 0, "filter9_vehicle": 0, "filter10_traffic_light": 0,
        "filter11_background": 0, "filter12_structure": 0, "passed": 0
    }

    per_color_stats = {c: {"det": 0, "tp": 0, "fp": 0} for c in color_channels}

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_det = 0

    print("=" * 80)
    print(f"RUNNING {exp_key}: {exp_desc}")
    print(f"Channels: {color_channels} | Morphology: {morph_tag}")
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

        # Base masks
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

        veg = _create_vegetation_mask(hsv)
        inv_veg = cv2.bitwise_not(veg)
        fr_sub = cv2.bitwise_and(fr, inv_veg)
        fb_sub = cv2.bitwise_and(fb, inv_veg)
        fy_sub = cv2.bitwise_and(fy, inv_veg)

        color_mask_dict = {
            "red": fr_sub,
            "blue": fb_sub,
            "yellow": fy_sub,
        }

        # Process each color independently and extract candidates
        img_candidates: List[Dict[str, Any]] = []

        for c_name in color_channels:
            raw_c_mask = color_mask_dict[c_name]
            cl_c_mask = apply_morphology(raw_c_mask, kernel_size, close_iters, open_iters)

            # Save mask for inspection
            stem = Path(img_name).stem
            cv2.imwrite(str(masks_dir / f"{stem}_{c_name}_morph_mask.png"), cl_c_mask)

            aux = {
                "mask_red": fr_sub,
                "mask_blue": fb_sub,
                "mask_yellow": fy_sub,
                "mask_white": hw,
                "mask_combined": raw_c_mask,
            }

            c_candidates, c_rejections = extract_candidates_with_provenance_and_filter_tracking(
                cl_c_mask, cfg, aux, (h, w), enh, color_channel=c_name, morph_setting=morph_tag
            )

            for k, val in c_rejections.items():
                aggregate_rejections[k] += val

            per_color_stats[c_name]["det"] += len(c_candidates)
            img_candidates.extend(c_candidates)

        # Ground truth evaluation
        gt_boxes = all_gt.get(img_name, [])
        eval_res = evaluate_one_to_one_provenance(img_candidates, gt_boxes, iou_threshold=iou_thresh)

        total_gt += len(gt_boxes)
        total_det += len(img_candidates)
        total_tp += eval_res["tp"]
        total_fp += eval_res["fp"]
        total_fn += eval_res["fn"]

        # Track per-color TP and FP
        for match in eval_res["matches"]:
            c_tag = match["det_candidate"]["color_channel"]
            per_color_stats[c_tag]["tp"] += 1

        for di in eval_res["unmatched_det_indices"]:
            c_tag = img_candidates[di]["color_channel"]
            per_color_stats[c_tag]["fp"] += 1

        # Draw overlay
        overlay = draw_annotated_overlay(enh, img_candidates, gt_boxes, eval_res, title=f"{exp_key}: {morph_tag}")
        cv2.imwrite(str(overlays_dir / f"{Path(img_name).stem}_overlay.png"), overlay)

        per_image_rows.append({
            "image": img_name,
            "gt_signs": len(gt_boxes),
            "detected": len(img_candidates),
            "tp": eval_res["tp"],
            "fp": eval_res["fp"],
            "fn": eval_res["fn"],
            "precision": f"{eval_res['precision']:.4f}",
            "recall": f"{eval_res['recall']:.4f}",
            "f1": f"{eval_res['f1']:.4f}",
        })

        for di, cand in enumerate(img_candidates):
            candidates_provenance_rows.append({
                "image": img_name,
                "det_id": di,
                "x1": cand["x1"],
                "y1": cand["y1"],
                "x2": cand["x2"],
                "y2": cand["y2"],
                "w": cand["w"],
                "h": cand["h"],
                "area": cand["area"],
                "color_channel": cand["color_channel"],
                "morph_setting": cand["morph_setting"],
            })

    # Save per-image CSV
    with open(exp_dir / f"{exp_key}_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        w.writeheader()
        w.writerows(per_image_rows)

    # Save candidate provenance CSV
    with open(exp_dir / f"{exp_key}_candidates_provenance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(candidates_provenance_rows[0].keys()))
        w.writeheader()
        w.writerows(candidates_provenance_rows)

    # Save filter rejection breakdown CSV
    with open(exp_dir / f"{exp_key}_filter_rejections.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Filter", "Count_Rejected"])
        for k, v in aggregate_rejections.items():
            w.writerow([k, v])

    overall_p = total_tp / max(1, total_tp + total_fp)
    overall_r = total_tp / max(1, total_tp + total_fn)
    overall_f1 = 2 * overall_p * overall_r / max(1e-8, overall_p + overall_r)

    print(f"[{exp_key} Overall] GT={total_gt} Det={total_det} TP={total_tp} FP={total_fp} FN={total_fn} P={overall_p:.4f} R={overall_r:.4f} F1={overall_f1:.4f}")
    for c_name, st in per_color_stats.items():
        print(f"  Per-color [{c_name}]: Det={st['det']} TP={st['tp']} FP={st['fp']}")
    print(f"  Filter Rejections: raw={aggregate_rejections['raw_contours']}, area={aggregate_rejections['filter1_area']}, dim={aggregate_rejections['filter2_dimension']}, asp={aggregate_rejections['filter3_aspect_ratio']}, pos={aggregate_rejections['filter4_spatial_position']}, veh={aggregate_rejections['filter9_vehicle']}, bg={aggregate_rejections['filter11_background']}, struct={aggregate_rejections['filter12_structure']}, passed={aggregate_rejections['passed']}\n")

    return {
        "exp_key": exp_key,
        "description": exp_desc,
        "channels": "+".join(color_channels),
        "morphology": morph_tag,
        "total_gt": total_gt,
        "total_det": total_det,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": overall_p,
        "recall": overall_r,
        "f1": overall_f1,
        "per_color": per_color_stats,
        "rejections": aggregate_rejections,
    }


def main():
    root = Path(__file__).resolve().parent
    gtsdb_dir = root / "data" / "TrainIJCNN2013" / "TrainIJCNN2013"
    gt_file = gtsdb_dir / "gt.txt"
    out_dir = root / "experiments_color_morphology"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_images = [f"{i:05d}.ppm" for i in range(20)]
    all_gt, val_stats = load_gtsdb_annotations(gt_file, gtsdb_dir)

    exp_defs = [
        {
            "key": "E1_Red_Only",
            "desc": "Red-only reference (k=5, close=3, open=1)",
            "channels": ["red"],
            "kernel_size": 5, "close_iters": 3, "open_iters": 1,
        },
        {
            "key": "E2_Red_Blue_Independent",
            "desc": "Red and Blue masks processed independently (k=5, close=3, open=1), combine boxes",
            "channels": ["red", "blue"],
            "kernel_size": 5, "close_iters": 3, "open_iters": 1,
        },
        {
            "key": "E3_Red_Blue_Yellow_Independent",
            "desc": "Red, Blue, and Yellow masks processed independently (k=5, close=3, open=1), combine boxes",
            "channels": ["red", "blue", "yellow"],
            "kernel_size": 5, "close_iters": 3, "open_iters": 1,
        },
        {
            "key": "E4_Independent_Small_Closing_k3",
            "desc": "Red, Blue, and Yellow masks processed independently with 1 close iter (k=3, close=1, open=1), combine boxes",
            "channels": ["red", "blue", "yellow"],
            "kernel_size": 3, "close_iters": 1, "open_iters": 1,
        },
    ]

    all_summaries = []
    for exp in exp_defs:
        summary = run_experiment(
            exp_key=exp["key"],
            exp_desc=exp["desc"],
            color_channels=exp["channels"],
            kernel_size=exp["kernel_size"],
            close_iters=exp["close_iters"],
            open_iters=exp["open_iters"],
            sample_images=sample_images,
            all_gt=all_gt,
            gtsdb_dir=gtsdb_dir,
            out_dir=out_dir,
            iou_thresh=0.5,
        )
        all_summaries.append(summary)

    # Save summary CSV
    summary_csv = out_dir / "experiments_e1_to_e4_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        fields = ["exp_key", "description", "channels", "morphology", "total_gt", "total_det", "tp", "fp", "fn", "precision", "recall", "f1"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in all_summaries:
            row = {k: s[k] for k in fields}
            row["precision"] = f"{s['precision']:.4f}"
            row["recall"] = f"{s['recall']:.4f}"
            row["f1"] = f"{s['f1']:.4f}"
            w.writerow(row)

    print("\n" + "=" * 105)
    print("FINAL EXPERIMENTS E1 - E4 SUMMARY (One-to-One IoU >= 0.5)")
    print("=" * 105)
    print(f"{'Experiment':<32} | {'Channels':<18} | {'Morphology':<18} | {'Det':<5} | {'TP':<3} | {'FP':<5} | {'FN':<3} | {'Prec':<7} | {'Rec':<7} | {'F1':<7}")
    print("-" * 105)
    for s in all_summaries:
        print(f"{s['exp_key']:<32} | {s['channels']:<18} | {s['morphology']:<18} | {s['total_det']:<5d} | {s['tp']:<3d} | {s['fp']:<5d} | {s['fn']:<3d} | {s['precision']:<7.4f} | {s['recall']:<7.4f} | {s['f1']:<7.4f}")
    print("=" * 105)
    print(f"Summary CSV written to: {summary_csv}\n")


if __name__ == "__main__":
    main()
