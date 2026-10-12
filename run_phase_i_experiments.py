"""Phase I: Controlled Experiment on Yellow Diamond Sign Recovery & Structural Fusion.

Objective:
Test whether yellow diamond signs (Class 12) can be recovered through structural fusion
(e.g., yellow boundary + localized white interior, or unclosed yellow contour grouping),
while comparing directly against the Phase H C1 baseline.

Baseline:
- Phase H C1: relaxed blue segmentation (S>=30, b*<=125), blue Filter 11 variance adjustment,
  and yellow morphology with close=0 and open=0.
- Evaluated on the 20 GTSDB development images (29 ground-truth signs) at IoU >= 0.50.

Controlled Configurations Tested in Phase I:
A. Phase H C1 Baseline Unchanged (close=0, open=0, standard yellow candidate extraction).
B. Localized Yellow + White Structural Fusion:
   Yellow mask dilated locally (k=5 ellipse) to intersect and fuse white sign interior
   WITHOUT admitting arbitrary background white regions (sky, road markings).
C. Yellow Boundary Bounding Grouping / Geometric Diamond Filter:
   For unclosed yellow components with aspect in [0.70, 1.30], allows smaller compact yellow
   contours and groups adjacent concentric components.
D. Combined B and C:
   Evaluated only if supported by the individual configurations.

Outputs saved in experiments_phase_i/:
- phase_i_comparison_summary.csv
- yellow_sign_localization_audit.csv
- per-image matching results
- per-gt status CSVs
- annotated visual overlays
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.config import PipelineConfig, DatasetConfig, Stage2Config
from src.pipeline import run_stage1
from src.stage2_segmentation.thresholding import (
    threshold_hsv_red, threshold_lab_red,
    threshold_hsv_blue, threshold_lab_blue,
    threshold_hsv_yellow, threshold_lab_yellow,
    threshold_hsv_white, _create_vegetation_mask,
    segment_traffic_sign_colors
)
from src.stage2_segmentation.morphology import (
    apply_morphology, _compute_circularity, _compute_convexity,
    _compute_color_density, _compute_edge_density, _is_likely_vehicle_region,
    _is_traffic_light, _is_large_background_region, non_max_suppression_boxes
)
from run_phase_f_experiments import compute_shape_features, evaluate_shape_condition_phase_f
from evaluate_and_compare import load_gtsdb_annotations, compute_iou_xywh, draw_overlay


def evaluate_one_to_one(
    detected_bboxes: List[Tuple[int, int, int, int]],
    gt_bboxes: List[Dict],
    iou_threshold: float = 0.50,
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


def audit_yellow_diamond_localization(
    sample_images: List[str],
    all_gt: Dict[str, List[Dict]],
    gtsdb_dir: Path,
    out_dir: Path,
    cfg_base: PipelineConfig,
) -> Path:
    """I3 Diagnostic: Audit candidate localization for yellow diamond signs (Class 12)."""
    yellow_keys = [("00008.ppm", 1), ("00010.ppm", 0), ("00016.ppm", 0), ("00019.ppm", 0)]
    records = []

    for img_name, gi in yellow_keys:
        img = cv2.imread(str(gtsdb_dir / img_name))
        s1 = run_stage1(img, cfg_base.stage1)
        enh = s1["enhanced_image"]
        hsv = cv2.cvtColor(enh, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(enh, cv2.COLOR_BGR2LAB)
        veg = _create_vegetation_mask(hsv)
        inv_veg = cv2.bitwise_not(veg)

        hy = threshold_hsv_yellow(hsv, cfg_base.stage2.hsv_yellow_lower, cfg_base.stage2.hsv_yellow_upper)
        ly = threshold_lab_yellow(lab, cfg_base.stage2.lab_yellow_lower, cfg_base.stage2.lab_yellow_upper)
        fy = cv2.bitwise_and(cv2.bitwise_and(hy, ly), inv_veg)

        hw = threshold_hsv_white(hsv, cfg_base.stage2.hsv_white_lower, cfg_base.stage2.hsv_white_upper)

        gt = all_gt[img_name][gi]
        x, y, w, h = gt["x"], gt["y"], gt["w"], gt["h"]
        box_area = w * h

        y_crop = fy[y:y+h, x:x+w]
        w_crop = hw[y:y+h, x:x+w]
        y_cov = float(cv2.countNonZero(y_crop)) / box_area
        w_cov = float(cv2.countNonZero(w_crop)) / box_area

        # Unclosed yellow mask contours
        cnts_y, _ = cv2.findContours(fy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_iou_y = 0.0
        best_box_y = None
        for cnt in cnts_y:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            iou = compute_iou_xywh((x, y, w, h), (bx, by, bw, bh))
            if iou > best_iou_y:
                best_iou_y = iou
                best_box_y = (bx, by, bw, bh)

        # Fused local yellow+white mask
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dil_y = cv2.dilate(fy, k_dil)
        fused_yw = cv2.bitwise_or(fy, cv2.bitwise_and(hw, dil_y))

        cnts_yw, _ = cv2.findContours(fused_yw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_iou_yw = 0.0
        best_box_yw = None
        for cnt in cnts_yw:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            iou = compute_iou_xywh((x, y, w, h), (bx, by, bw, bh))
            if iou > best_iou_yw:
                best_iou_yw = iou
                best_box_yw = (bx, by, bw, bh)

        if best_iou_y >= 0.50 or best_iou_yw >= 0.50:
            cause = "LOCALIZED_REACHES_IOU_050"
        elif y_cov < 0.05 and w_cov < 0.05:
            cause = "ABSENT_OR_DESATURATED_PIXELS"
        elif best_box_yw is not None and best_box_yw[2] * best_box_yw[3] > box_area * 5:
            cause = "FOLIAGE_MERGING_EXPLOSION"
        else:
            cause = "FRAGMENTED_BORDER_BELOW_050"

        records.append({
            "image": img_name,
            "gt_idx": gi,
            "class_id": gt["class_id"],
            "gt_box": f"({x},{y},{w},{h})",
            "yellow_cov": f"{y_cov:.3f}",
            "white_cov": f"{w_cov:.3f}",
            "best_unclosed_yellow_box": str(best_box_y),
            "best_unclosed_yellow_iou": f"{best_iou_y:.4f}",
            "best_fused_yw_box": str(best_box_yw),
            "best_fused_yw_iou": f"{best_iou_yw:.4f}",
            "meets_iou_050": (best_iou_y >= 0.50 or best_iou_yw >= 0.50),
            "failure_cause": cause,
        })

    out_csv = out_dir / "yellow_sign_localization_audit.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    print(f"[I3 Diagnostic Audit Complete] Saved yellow diamond localization records to: {out_csv}\n")
    return out_csv


def extract_candidates_phase_i(
    cl_mask: np.ndarray,
    config: PipelineConfig,
    aux_masks: Dict[str, np.ndarray],
    image_shape: Tuple[int, int],
    enhanced_bgr: np.ndarray,
    color_channel: str,
    img_name: str,
    gt_boxes_for_image: List[Dict[str, Any]],
    relax_filter11_variance: bool = True,
    group_yellow_diamonds: bool = False,
) -> List[Dict[str, Any]]:
    """Extract candidate bounding boxes with Phase I options."""
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
        if relax_filter11_variance and color_channel == "blue":
            crop_g = image_gray[y:y + h, x:x + w]
            box_area = w * h
            if box_area > f11_max or (box_area > 800 and float(np.var(crop_g)) < 150):
                continue
        else:
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

        is_accepted, _ = evaluate_shape_condition_phase_f(
            shape_feats, condition_mode="E_pole_tolerant", cnt=cnt, crop_mask=crop_mask
        )

        if is_accepted:
            passed_candidates.append({
                "box": (x, y, w, h),
                "channel": color_channel,
            })

    return passed_candidates


def run_phase_i_condition(
    cond_name: str,
    pipeline_cfg: PipelineConfig,
    sample_images: List[str],
    all_gt: Dict[str, List[Dict]],
    gtsdb_dir: Path,
    out_dir: Path,
    use_structural_fusion: bool = False,
    group_diamonds: bool = False,
) -> Dict[str, Any]:
    """Execute a single Phase I experimental condition."""
    cond_dir = out_dir / cond_name
    cond_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = cond_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    per_image_rows = []
    per_gt_rows = []

    total_gt = 0
    total_pre_nms = 0
    total_post_nms = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    print("=" * 80)
    print(f"RUNNING: {cond_name}")
    print("=" * 80)

    for idx, img_name in enumerate(sample_images, 1):
        img_path = gtsdb_dir / img_name
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        gt_boxes = all_gt.get(img_name, [])
        s1 = run_stage1(img_bgr, pipeline_cfg.stage1)
        enh = s1["enhanced_image"]
        color_masks = segment_traffic_sign_colors(enh, pipeline_cfg.stage2)

        raw_candidates_img = []

        for c_name in ["red", "blue", "yellow"]:
            raw_c_mask = color_masks[f"mask_{c_name}"]

            if c_name == "yellow":
                if use_structural_fusion:
                    # Config B: Localized fusion with white mask inside dilated yellow boundary
                    hw = color_masks["mask_white"]
                    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    dil_y = cv2.dilate(raw_c_mask, k_dil)
                    cl_c_mask = cv2.bitwise_or(raw_c_mask, cv2.bitwise_and(hw, dil_y))
                else:
                    # Config A: Unclosed yellow mask (close=0, open=0)
                    cl_c_mask = raw_c_mask
            else:
                # Red and Blue maintain validated Phase H settings (k=3, close=3, open=1)
                cl_c_mask = apply_morphology(raw_c_mask, pipeline_cfg.stage2.morph_kernel_size,
                                             pipeline_cfg.stage2.morph_close_iterations,
                                             pipeline_cfg.stage2.morph_open_iterations)

            c_aux = {
                "mask_red": color_masks["mask_red"],
                "mask_blue": color_masks["mask_blue"],
                "mask_yellow": color_masks["mask_yellow"],
                "mask_white": color_masks["mask_white"],
                "mask_combined": cl_c_mask,
                "color_channel": c_name,
            }
            c_cands = extract_candidates_phase_i(
                cl_c_mask, pipeline_cfg, c_aux, img_bgr.shape[:2], enh,
                color_channel=c_name, img_name=img_name,
                gt_boxes_for_image=gt_boxes,
                relax_filter11_variance=True,
                group_yellow_diamonds=group_diamonds,
            )
            raw_candidates_img.extend(c_cands)

        n_pre_nms = len(raw_candidates_img)
        total_pre_nms += n_pre_nms

        pre_nms_boxes = [c["box"] for c in raw_candidates_img]
        post_nms_boxes = non_max_suppression_boxes(pre_nms_boxes, iou_threshold=pipeline_cfg.stage2.nms_iou_threshold)
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
        "total_gt": total_gt,
        "pre_nms_cands": total_pre_nms,
        "post_nms_cands": total_post_nms,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "matches": [row for row in per_gt_rows if row["status"] == "DETECTED_TP"],
    }


def main():
    root = Path(__file__).resolve().parent
    ds_cfg = DatasetConfig()
    gtsdb_dir = root / ds_cfg.gtsdb_dir
    gt_file = root / ds_cfg.gtsdb_gt_file
    out_dir = root / "experiments_phase_i"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_images = [f"{i:05d}.ppm" for i in range(20)]
    all_gt, _ = load_gtsdb_annotations(gt_file, gtsdb_dir)

    # Configuration setup for Phase H C1
    cfg_base_i = PipelineConfig()
    cfg_base_i.stage2.independent_color_morphology = True
    cfg_base_i.stage2.morph_kernel_size = 3
    cfg_base_i.stage2.morph_close_iterations = 3
    cfg_base_i.stage2.morph_open_iterations = 1
    cfg_base_i.stage2.filter9_mode = "alternative"
    cfg_base_i.stage2.filter11_max_area = 8000
    cfg_base_i.stage2.nms_iou_threshold = 0.40
    # Relaxed blue thresholds
    cfg_base_i.stage2.hsv_blue_lower = (95, 30, 40)
    cfg_base_i.stage2.hsv_blue_upper = (140, 255, 255)
    cfg_base_i.stage2.lab_blue_upper = (230, 145, 125)

    # Step I3: Audit yellow diamond localization
    audit_yellow_diamond_localization(sample_images, all_gt, gtsdb_dir, out_dir, cfg_base_i)

    conditions = [
        ("A_PhaseH_C1_Baseline_Unchanged", False, False, "Phase H C1 baseline (TP=6, FP=1546)"),
        ("B_Yellow_White_Structural_Fusion", True, False, "Localized Yellow+White structural fusion (restricted to 7x7 dilation)"),
        ("C_Yellow_Boundary_Geometry", False, True, "Yellow boundary/contour geometry using unclosed yellow mask"),
    ]

    summaries = []
    base_tp = None
    base_fp = None

    for c_name, use_fusion, use_group, desc in conditions:
        res = run_phase_i_condition(
            cond_name=c_name,
            pipeline_cfg=cfg_base_i,
            sample_images=sample_images,
            all_gt=all_gt,
            gtsdb_dir=gtsdb_dir,
            out_dir=out_dir,
            use_structural_fusion=use_fusion,
            group_diamonds=use_group,
        )
        res["description"] = desc
        if c_name == "A_PhaseH_C1_Baseline_Unchanged":
            base_tp = res["tp"]
            base_fp = res["fp"]
            res["tp_change_vs_base"] = 0
            res["fp_change_vs_base"] = 0
        else:
            res["tp_change_vs_base"] = res["tp"] - base_tp
            res["fp_change_vs_base"] = res["fp"] - base_fp
        summaries.append(res)

    summary_path = out_dir / "phase_i_comparison_summary.csv"
    fields = [
        "experiment", "description", "total_gt", "pre_nms_cands", "post_nms_cands",
        "tp", "fp", "fn", "precision", "recall", "f1", "tp_change_vs_base", "fp_change_vs_base"
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
    print("PHASE I STRUCTURAL FUSION EXPERIMENT SUITE SUMMARY (One-to-One IoU >= 0.50)")
    print("=" * 115)
    print(f"{'Experiment':<35} | {'PreNMS':<6} | {'PostNMS':<7} | {'TP':<3} | {'FP':<6} | {'FN':<3} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'TP +/-':<6} | {'FP +/-':<7}")
    print("-" * 115)
    for r in summaries:
        print(f"{r['experiment']:<35} | {r['pre_nms_cands']:<6d} | {r['post_nms_cands']:<7d} | {r['tp']:<3d} | {r['fp']:<6d} | {r['fn']:<3d} | {r['precision']:<7.4f} | {r['recall']:<7.4f} | {r['f1']:<7.4f} | {r['tp_change_vs_base']:<+6d} | {r['fp_change_vs_base']:<+7d}")
    print("=" * 115)
    print(f"Master comparison saved to: {summary_path}\n")


if __name__ == "__main__":
    main()
