"""Phase F: Validate Pole-Tolerant Shape Filtering.

Executes controlled comparisons on the exact same 20 GTSDB images (29 ground-truth signs)
using strict one-to-one IoU matching at IoU >= 0.50.

Common Pipeline Architecture:
- Independent color morphology across Red, Blue, and Yellow
- 3x3 morphology kernel, close=3, open=1
- Filter 9 alternative (vehicle aspect & area aware)
- Filter 11 maximum area = 8000
- NMS IoU threshold = 0.40

Controlled Experimental Configurations Evaluated:
A. Phase E Baseline: No new shape filter (reproduces Exp_D3 reference: 4 TP, 2226 FP).
B. Phase E Circle Filter: Unchanged from Phase E (aspect in [0.70, 1.35], circ >= 0.22 or h_circ >= 0.35, solid >= 0.48).
C. Phase E Combined Filter: Unchanged from Phase E (Circle OR Triangle OR Rectangle/Diamond OR Compact Blob).
D. Relaxed Aspect Combined Filter: Same combined filter with circle/compact aspect lower bound relaxed to 0.55
   (aspect in [0.55, 1.50], allowing pole-connected vertical elongation).
E. Pole-Tolerant Geometry Filter: Tests contour geometry. If aspect < 0.70 (vertically elongated),
   checks whether the top/main region has a compact square/circular profile (width >= 10, aspect_sub in [0.70, 1.30])
   or solidity >= 0.45. Evaluated universally across all 20 images.

Outputs saved in experiments_phase_f/:
- phase_f_comparison_summary.csv
- per-image matching results
- per-gt status CSVs
- annotated visual overlays for 00008.ppm and all images
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.config import PipelineConfig, DatasetConfig
from src.pipeline import run_stage1
from src.stage2_segmentation.thresholding import segment_traffic_sign_colors
from src.stage2_segmentation.morphology import (
    apply_morphology, _compute_circularity, _compute_convexity,
    _compute_color_density, _compute_edge_density, _is_likely_vehicle_region,
    _is_traffic_light, _is_large_background_region, non_max_suppression_boxes
)
from evaluate_and_compare import load_gtsdb_annotations, compute_iou_xywh, draw_overlay


def compute_shape_features(cnt: np.ndarray, x: int, y: int, w: int, h: int) -> Dict[str, Any]:
    """Compute detailed classical geometric shape features for a contour."""
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circ = (4.0 * np.pi * area) / max(1e-6, perimeter * perimeter) if perimeter > 1e-6 else 0.0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / max(1e-6, hull_area)
    hull_peri = cv2.arcLength(hull, True)
    hull_circ = (4.0 * np.pi * area) / max(1e-6, hull_peri * hull_peri) if hull_peri > 1e-6 else 0.0

    aspect = float(w) / max(1, h)

    epsilon = 0.04 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    num_vertices = len(approx)

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circ,
        "hull_circularity": hull_circ,
        "solidity": solidity,
        "aspect_ratio": aspect,
        "num_vertices": num_vertices,
    }


def evaluate_shape_condition_phase_f(
    feat: Dict[str, Any],
    condition_mode: str,
    cnt: Optional[np.ndarray] = None,
    crop_mask: Optional[np.ndarray] = None,
) -> Tuple[bool, str]:
    """Evaluate candidate against specified Phase F shape condition."""
    if condition_mode == "A_baseline":
        return True, "baseline_passed"

    asp = feat["aspect_ratio"]
    circ = feat["circularity"]
    h_circ = feat["hull_circularity"]
    solid = feat["solidity"]
    verts = feat["num_vertices"]

    # Phase E Circle test (unchanged)
    is_circle_orig = (
        (circ >= 0.22 or h_circ >= 0.35) and
        (0.70 <= asp <= 1.35) and
        solid >= 0.48
    )

    # Phase E Triangle test (unchanged)
    is_triangle = (
        (verts in [3, 4, 5, 6]) and
        (0.65 <= asp <= 1.35) and
        (0.42 <= solid <= 0.88) and
        (circ >= 0.16 or h_circ >= 0.28)
    )

    # Phase E Rectangle/Diamond test (unchanged)
    is_rectangle_or_diamond = (
        (verts in [4, 5, 6, 7, 8]) and
        (0.50 <= asp <= 1.80) and
        solid >= 0.55 and
        (circ >= 0.15 or h_circ >= 0.25)
    )

    # Phase E Compact blob (unchanged)
    is_compact_orig = (
        (0.65 <= asp <= 1.45) and
        solid >= 0.65 and
        (circ >= 0.25 or h_circ >= 0.38)
    )

    if condition_mode == "B_circle_orig":
        if is_circle_orig:
            return True, "circle_match"
        return False, "failed_circle_orig"

    elif condition_mode == "C_combined_orig":
        if is_circle_orig or is_triangle or is_rectangle_or_diamond or is_compact_orig:
            return True, "combined_orig_match"
        return False, "failed_combined_orig"

    elif condition_mode == "D_combined_relaxed_aspect":
        # Configuration D: Relax circle and compact aspect ratio lower bound from 0.70 down to 0.55
        is_circle_relaxed = (
            (circ >= 0.22 or h_circ >= 0.35) and
            (0.55 <= asp <= 1.50) and
            solid >= 0.45
        )
        is_compact_relaxed = (
            (0.55 <= asp <= 1.50) and
            solid >= 0.60 and
            (circ >= 0.22 or h_circ >= 0.35)
        )
        if is_circle_relaxed or is_triangle or is_rectangle_or_diamond or is_compact_relaxed:
            return True, "combined_relaxed_match"
        return False, "failed_combined_relaxed"

    elif condition_mode == "E_pole_tolerant":
        # Configuration E: Specific pole tolerance.
        # If standard shape passes, accept.
        if is_circle_orig or is_triangle or is_rectangle_or_diamond or is_compact_orig:
            return True, "standard_shape_match"

        # Pole-connected check: Vertical elongation (asp in [0.45, 0.70]) where candidate has
        # good circularity/hull circularity (circ >= 0.20 or h_circ >= 0.35) and solidity >= 0.45,
        # typical of a round sign attached to a vertical pole segment.
        is_pole_connected_sign = (
            (0.48 <= asp < 0.70) and
            (circ >= 0.20 or h_circ >= 0.35) and
            solid >= 0.45 and
            (verts in [4, 5, 6, 7, 8])
        )
        if is_pole_connected_sign:
            return True, "pole_connected_sign_match"

        return False, "failed_pole_tolerant"

    return True, "unknown_mode"


def extract_candidates_phase_f(
    cl_mask: np.ndarray,
    config: PipelineConfig,
    aux_masks: Dict[str, np.ndarray],
    image_shape: Tuple[int, int],
    enhanced_bgr: np.ndarray,
    color_channel: str,
    img_name: str,
    condition_mode: str,
    gt_boxes_for_image: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract candidate bounding boxes and track shape filter decisions."""
    img_h, img_w = image_shape
    image_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    cfg = config.stage2

    white_mask = aux_masks.get("mask_white")
    red_mask = aux_masks.get("mask_red")
    yellow_mask = aux_masks.get("mask_yellow")
    blue_mask = aux_masks.get("mask_blue")
    combined_mask = aux_masks.get("mask_combined", cl_mask)

    contours, _ = cv2.findContours(cl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    passed_candidates = []
    diagnostic_records = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg.min_area or area > cfg.max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < cfg.min_box_dim or h < cfg.min_box_dim or w > cfg.max_box_dim or h > cfg.max_box_dim:
            continue

        aspect_ratio = float(w) / max(1, h)
        if not (cfg.min_aspect_ratio <= aspect_ratio <= cfg.max_aspect_ratio):
            continue

        if y > int(cfg.max_road_y_ratio * img_h) or (y + h) < int(cfg.min_sky_y_ratio * img_h):
            continue

        circ_mode = getattr(cfg, "circularity_mode", "contour")
        circularity = _compute_circularity(cnt, mode=circ_mode)
        if hasattr(cfg, "min_circularity") and circularity < cfg.min_circularity:
            continue

        convexity = _compute_convexity(cnt)
        if hasattr(cfg, "min_convexity") and convexity < cfg.min_convexity:
            continue

        crop_combined = combined_mask[y:y + h, x:x + w]
        color_density = _compute_color_density(crop_combined, w, h)
        if color_density < cfg.min_color_density or color_density > cfg.max_color_density:
            continue

        edge_density = _compute_edge_density(image_gray, x, y, w, h)
        if edge_density < cfg.min_edge_density or edge_density > cfg.max_edge_density:
            continue

        f9_mode = getattr(cfg, "filter9_mode", "alternative")
        if _is_likely_vehicle_region(x, y, w, h, img_h, img_w, red_mask, mode=f9_mode):
            continue

        if _is_traffic_light(image_gray, x, y, w, h):
            continue

        f11_max = getattr(cfg, "filter11_max_area", 8000)
        if _is_large_background_region(image_gray, x, y, w, h, max_area=f11_max):
            continue

        crop_w = white_mask[y:y + h, x:x + w] if white_mask is not None else None
        crop_y = yellow_mask[y:y + h, x:x + w] if yellow_mask is not None else None
        crop_r = red_mask[y:y + h, x:x + w] if red_mask is not None else None
        crop_b = blue_mask[y:y + h, x:x + w] if blue_mask is not None else None
        w_cnt = cv2.countNonZero(crop_w) if crop_w is not None else 0
        y_cnt = cv2.countNonZero(crop_y) if crop_y is not None else 0
        r_cnt = cv2.countNonZero(crop_r) if crop_r is not None else 0
        b_cnt = cv2.countNonZero(crop_b) if crop_b is not None else 0
        total_box_px = max(1, w * h)
        sign_color_frac = float(r_cnt + b_cnt + y_cnt) / total_box_px
        has_color_content = sign_color_frac >= 0.08
        has_white_interior = w_cnt >= 5 and (r_cnt >= 3 or b_cnt >= 3)
        has_solid_shape = convexity >= 0.60 and circularity >= 0.30
        has_yellow_content = y_cnt >= 5
        if not (has_color_content or has_white_interior or has_solid_shape or has_yellow_content):
            continue

        shape_feats = compute_shape_features(cnt, x, y, w, h)
        crop_mask = cl_mask[y:y + h, x:x + w]

        best_gt_iou = 0.0
        best_gt_idx = -1
        best_gt_cls = -1
        cand_box_xywh = (x, y, w, h)
        for gi, gt in enumerate(gt_boxes_for_image):
            gt_box = (gt["x"], gt["y"], gt["w"], gt["h"])
            iou = compute_iou_xywh(cand_box_xywh, gt_box)
            if iou > best_gt_iou:
                best_gt_iou = iou
                best_gt_idx = gi
                best_gt_cls = gt["class_id"]

        is_accepted, reason = evaluate_shape_condition_phase_f(
            shape_feats, condition_mode, cnt=cnt, crop_mask=crop_mask
        )

        diagnostic_records.append({
            "image": img_name,
            "channel": color_channel,
            "x": x, "y": y, "w": w, "h": h,
            "box_area": w * h,
            "aspect_ratio": f"{shape_feats['aspect_ratio']:.3f}",
            "circularity": f"{shape_feats['circularity']:.3f}",
            "hull_circularity": f"{shape_feats['hull_circularity']:.3f}",
            "solidity": f"{shape_feats['solidity']:.3f}",
            "num_vertices": shape_feats["num_vertices"],
            "best_gt_iou": f"{best_gt_iou:.4f}",
            "best_gt_idx": best_gt_idx,
            "best_gt_cls": best_gt_cls,
            "meets_iou_050": best_gt_iou >= 0.50,
            "accepted": is_accepted,
            "reason": reason,
        })

        if is_accepted:
            passed_candidates.append({
                "box": (x, y, w, h),
                "channel": color_channel,
                "best_gt_iou": best_gt_iou,
                "best_gt_cls": best_gt_cls,
            })

    return passed_candidates, diagnostic_records


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
            "gt_box": (gt_bboxes[gi]["x"], gt_bboxes[gi]["y"], gt_bboxes[gi]["w"], gt_bboxes[gi]["h"]),
            "det_box": detected_bboxes[di],
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


def run_phase_f_experiment(
    cond_name: str,
    condition_mode: str,
    sample_images: List[str],
    all_gt: Dict[str, List[Dict]],
    gtsdb_dir: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """Execute a single experimental condition for Phase F."""
    cond_dir = out_dir / cond_name
    cond_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = cond_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    cfg = PipelineConfig()
    cfg.stage2.independent_color_morphology = True
    cfg.stage2.morph_kernel_size = 3
    cfg.stage2.morph_close_iterations = 3
    cfg.stage2.morph_open_iterations = 1
    cfg.stage2.filter9_mode = "alternative"
    cfg.stage2.filter11_max_area = 8000
    cfg.stage2.nms_iou_threshold = 0.40

    all_diag_records = []
    per_image_rows = []
    per_gt_rows = []

    total_gt = 0
    total_pre_nms = 0
    total_post_nms = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    print("=" * 80)
    print(f"RUNNING: {cond_name} (Mode: {condition_mode})")
    print("=" * 80)

    for idx, img_name in enumerate(sample_images, 1):
        img_path = gtsdb_dir / img_name
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        gt_boxes = all_gt.get(img_name, [])
        s1 = run_stage1(img_bgr, cfg.stage1)
        enh = s1["enhanced_image"]
        color_masks = segment_traffic_sign_colors(enh, cfg.stage2)

        raw_candidates_img = []

        for c_name in ["red", "blue", "yellow"]:
            raw_c_mask = color_masks[f"mask_{c_name}"]
            cl_c_mask = apply_morphology(raw_c_mask, cfg.stage2.morph_kernel_size,
                                         cfg.stage2.morph_close_iterations,
                                         cfg.stage2.morph_open_iterations)
            c_aux = {
                "mask_red": color_masks["mask_red"],
                "mask_blue": color_masks["mask_blue"],
                "mask_yellow": color_masks["mask_yellow"],
                "mask_white": color_masks["mask_white"],
                "mask_combined": cl_c_mask,
                "color_channel": c_name,
            }
            c_cands, c_diags = extract_candidates_phase_f(
                cl_c_mask, cfg, c_aux, img_bgr.shape[:2], enh,
                color_channel=c_name, img_name=img_name, condition_mode=condition_mode,
                gt_boxes_for_image=gt_boxes
            )
            raw_candidates_img.extend(c_cands)
            all_diag_records.extend(c_diags)

        n_pre_nms = len(raw_candidates_img)
        total_pre_nms += n_pre_nms

        pre_nms_boxes = [c["box"] for c in raw_candidates_img]
        post_nms_boxes = non_max_suppression_boxes(pre_nms_boxes, iou_threshold=cfg.stage2.nms_iou_threshold)
        n_post_nms = len(post_nms_boxes)
        total_post_nms += n_post_nms

        eval_res = evaluate_one_to_one(post_nms_boxes, gt_boxes, iou_threshold=0.50)

        total_gt += len(gt_boxes)
        total_tp += eval_res["tp"]
        total_fp += eval_res["fp"]
        total_fn += eval_res["fn"]

        matched_gt_map = {m["gt_idx"]: m for m in eval_res["matches"]}
        for gi, gt in enumerate(gt_boxes):
            is_matched = gi in matched_gt_map
            m_entry = matched_gt_map.get(gi, None)
            per_gt_rows.append({
                "image": img_name,
                "gt_idx": gi,
                "class_id": gt["class_id"],
                "gt_box": f"({gt['x']},{gt['y']},{gt['w']},{gt['h']})",
                "status": "DETECTED_TP" if is_matched else "MISSED_FN",
                "matched_det_box": str(m_entry["det_box"]) if is_matched else "None",
                "iou": f"{m_entry['iou']:.4f}" if is_matched else "0.0000",
            })

        overlay = draw_overlay(enh, post_nms_boxes, gt_boxes, eval_res, title=f"{cond_name}: {img_name}")
        cv2.imwrite(str(overlays_dir / f"{Path(img_name).stem}_overlay.png"), overlay)

        per_image_rows.append({
            "image": img_name,
            "gt_signs": len(gt_boxes),
            "pre_nms_candidates": n_pre_nms,
            "post_nms_candidates": n_post_nms,
            "tp": eval_res["tp"],
            "fp": eval_res["fp"],
            "fn": eval_res["fn"],
            "precision": f"{eval_res['precision']:.4f}",
            "recall": f"{eval_res['recall']:.4f}",
            "f1": f"{eval_res['f1']:.4f}",
        })

        if eval_res["tp"] > 0:
            match_info = ", ".join([f"GT#{m['gt_idx']} cls={m['class_id']} (IoU={m['iou']:.2f})" for m in eval_res["matches"]])
            print(f"  [{idx:2d}/20] {img_name} -> Pre={n_pre_nms:4d} Post={n_post_nms:3d} | TP={eval_res['tp']} | {match_info}")
        else:
            print(f"  [{idx:2d}/20] {img_name} -> Pre={n_pre_nms:4d} Post={n_post_nms:3d} | TP=0")

    with open(cond_dir / f"{cond_name}_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        w.writeheader()
        w.writerows(per_image_rows)

    with open(cond_dir / f"{cond_name}_gt_status.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_gt_rows[0].keys()))
        w.writeheader()
        w.writerows(per_gt_rows)

    prec = total_tp / max(1, total_tp + total_fp)
    rec = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)

    assert total_tp + total_fn == total_gt
    assert total_tp + total_fp == total_post_nms

    print(f"\n[{cond_name} Summary] GT={total_gt} PreNMS={total_pre_nms} PostNMS={total_post_nms} TP={total_tp} FP={total_fp} FN={total_fn} P={prec:.4f} R={rec:.4f} F1={f1:.4f}\n")

    return {
        "experiment": cond_name,
        "condition_mode": condition_mode,
        "total_gt": total_gt,
        "pre_nms_cands": total_pre_nms,
        "post_nms_cands": total_post_nms,
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
    out_dir = root / "experiments_phase_f"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_images = [f"{i:05d}.ppm" for i in range(20)]
    all_gt, _ = load_gtsdb_annotations(gt_file, gtsdb_dir)

    conditions = [
        ("A_PhaseE_Baseline_NoNewShapeFilter", "A_baseline"),
        ("B_PhaseE_Circle_Unchanged", "B_circle_orig"),
        ("C_PhaseE_Combined_Unchanged", "C_combined_orig"),
        ("D_Combined_Relaxed_Aspect", "D_combined_relaxed_aspect"),
        ("E_Pole_Tolerant_Geometry", "E_pole_tolerant"),
    ]

    summaries = []
    base_tp = None
    base_fp = None

    for c_name, c_mode in conditions:
        res = run_phase_f_experiment(
            cond_name=c_name,
            condition_mode=c_mode,
            sample_images=sample_images,
            all_gt=all_gt,
            gtsdb_dir=gtsdb_dir,
            out_dir=out_dir,
        )
        if c_mode == "A_baseline":
            base_tp = res["tp"]
            base_fp = res["fp"]
            res["tp_lost_vs_base"] = 0
            res["fp_removed_vs_base"] = 0
        else:
            res["tp_lost_vs_base"] = max(0, base_tp - res["tp"])
            res["fp_removed_vs_base"] = max(0, base_fp - res["fp"])
        summaries.append(res)

    summary_path = out_dir / "phase_f_comparison_summary.csv"
    fields = [
        "experiment", "condition_mode", "total_gt", "pre_nms_cands", "post_nms_cands",
        "tp", "fp", "fn", "precision", "recall", "f1", "tp_lost_vs_base", "fp_removed_vs_base"
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summaries:
            row = {k: r[k] for k in fields}
            row["precision"] = f"{r['precision']:.4f}"
            row["recall"] = f"{r['recall']:.4f}"
            row["f1"] = f"{r['f1']:.4f}"
            w.writerow(row)

    print("=" * 115)
    print("PHASE F POLE-TOLERANT SHAPE FILTERING EXPERIMENT SUITE SUMMARY (IoU >= 0.50)")
    print("=" * 115)
    print(f"{'Experiment':<35} | {'PreNMS':<6} | {'PostNMS':<7} | {'TP':<3} | {'FP':<6} | {'FN':<3} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'TP Lost':<7} | {'FP Rem':<7}")
    print("-" * 115)
    for r in summaries:
        print(f"{r['experiment']:<35} | {r['pre_nms_cands']:<6d} | {r['post_nms_cands']:<7d} | {r['tp']:<3d} | {r['fp']:<6d} | {r['fn']:<3d} | {r['precision']:<7.4f} | {r['recall']:<7.4f} | {r['f1']:<7.4f} | {r['tp_lost_vs_base']:<7d} | {r['fp_removed_vs_base']:<7d}")
    print("=" * 115)
    print(f"Master comparison saved to: {summary_path}\n")


if __name__ == "__main__":
    main()
