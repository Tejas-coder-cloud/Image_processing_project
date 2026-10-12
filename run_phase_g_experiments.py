"""Phase G: Controlled Color-Segmentation & Diagnostic Experiment Suite.

Objective:
Determine why the 25 ground-truth signs missed by Phase F Configuration E were not detected,
and test whether Stage 2 color segmentation is the main bottleneck.

Baseline:
- 20 GTSDB development images (00000.ppm to 00019.ppm, 29 GT signs)
- Phase F Configuration E (Pole-Tolerant Geometry) as the baseline reference
- IoU >= 0.50 with strict one-to-one bipartite matching

Controlled Experiments:
A. Phase F Configuration E Baseline (Unmodified reference).
B. Red Thresholds Relaxed:
   - S_min: 50 -> 35
   - a*_min: 140 -> 128 (captures desaturated/shadowed signs like 00005, 00006, 00017)
C. Blue Thresholds Relaxed:
   - S_min: 40 -> 30, H_upper: 135 -> 140, b*_upper: 115 -> 125
   - (Plus Filter 11 variance condition adjustment for blue signs)
D. Yellow Thresholds Relaxed:
   - S_min: 50 -> 35, b*_min: 140 -> 130
E. Combined Best-Supported Color Settings:
   - Combines validated threshold adjustments from B, C, and D.

Outputs saved in experiments_phase_g/:
- phase_g_comparison_summary.csv
- missed_signs_audit_records.csv
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
    segment_traffic_sign_colors,
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


def audit_missed_signs(
    sample_images: List[str],
    all_gt: Dict[str, List[Dict]],
    gtsdb_dir: Path,
    out_dir: Path,
    cfg_base: PipelineConfig,
) -> Path:
    """Step 1: Perform comprehensive audit of the 25 missed signs from Phase F."""
    records = []
    detected_phase_f = {("00003.ppm", 1), ("00008.ppm", 0), ("00013.ppm", 0), ("00014.ppm", 0)}

    for img_name in sample_images:
        gts = all_gt.get(img_name, [])
        if not gts:
            continue
        img = cv2.imread(str(gtsdb_dir / img_name))
        s1 = run_stage1(img, cfg_base.stage1)
        enh = s1["enhanced_image"]
        hsv = cv2.cvtColor(enh, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(enh, cv2.COLOR_BGR2LAB)
        veg = _create_vegetation_mask(hsv)
        inv_veg = cv2.bitwise_not(veg)

        # Generate color masks
        hr = threshold_hsv_red(hsv, cfg_base.stage2.hsv_red_lower1, cfg_base.stage2.hsv_red_upper1,
                               cfg_base.stage2.hsv_red_lower2, cfg_base.stage2.hsv_red_upper2)
        lr = threshold_lab_red(lab, cfg_base.stage2.lab_red_lower, cfg_base.stage2.lab_red_upper)
        fr = cv2.bitwise_and(cv2.bitwise_and(hr, lr), inv_veg)
        mr = apply_morphology(fr, 3, 3, 1)

        hb = threshold_hsv_blue(hsv, cfg_base.stage2.hsv_blue_lower, cfg_base.stage2.hsv_blue_upper)
        lb = threshold_lab_blue(lab, cfg_base.stage2.lab_blue_lower, cfg_base.stage2.lab_blue_upper)
        fb = cv2.bitwise_and(cv2.bitwise_and(hb, lb), inv_veg)
        mb = apply_morphology(fb, 3, 3, 1)

        hy = threshold_hsv_yellow(hsv, cfg_base.stage2.hsv_yellow_lower, cfg_base.stage2.hsv_yellow_upper)
        ly = threshold_lab_yellow(lab, cfg_base.stage2.lab_yellow_lower, cfg_base.stage2.lab_yellow_upper)
        fy = cv2.bitwise_and(cv2.bitwise_and(hy, ly), inv_veg)
        my = apply_morphology(fy, 3, 3, 1)

        for gi, gt in enumerate(gts):
            x, y, w, h = gt["x"], gt["y"], gt["w"], gt["h"]
            cid = gt["class_id"]
            box_area = w * h

            r_pre = cv2.countNonZero(fr[y:y+h, x:x+w])
            r_post = cv2.countNonZero(mr[y:y+h, x:x+w])
            b_pre = cv2.countNonZero(fb[y:y+h, x:x+w])
            b_post = cv2.countNonZero(mb[y:y+h, x:x+w])
            y_pre = cv2.countNonZero(fy[y:y+h, x:x+w])
            y_post = cv2.countNonZero(my[y:y+h, x:x+w])

            # Definition: mask coverage = non-zero mask pixels inside GT bbox / total GT bbox area
            cov_r_pre = float(r_pre) / box_area
            cov_b_pre = float(b_pre) / box_area
            cov_y_pre = float(y_pre) / box_area

            # Find best overlapping component in morphology masks
            best_iou = 0.0
            best_c = "none"
            for c_name, m_mask in [("red", mr), ("blue", mb), ("yellow", my)]:
                for cnt in cv2.findContours(m_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    iou = compute_iou_xywh((x, y, w, h), (bx, by, bw, bh))
                    if iou > best_iou:
                        best_iou = iou
                        best_c = c_name

            # Determine primary failure stage
            if (img_name, gi) in detected_phase_f:
                failure_stage = "DETECTED_IN_PHASE_F"
            elif max(cov_r_pre, cov_b_pre, cov_y_pre) < 0.03:
                failure_stage = "ABSENT_OR_WEAK_COLOR_THRESHOLD"
            elif best_iou >= 0.50:
                failure_stage = "REJECTED_BY_DOWNSTREAM_FILTER"
            elif max(r_post, b_post, y_post) > box_area * 1.5:
                failure_stage = "CONNECTED_TO_BACKGROUND_CLUTTER"
            else:
                failure_stage = "FRAGMENTED_BELOW_IOU_050"

            records.append({
                "image": img_name,
                "gt_idx": gi,
                "class_id": cid,
                "gt_box": f"({x},{y},{w},{h})",
                "box_area": box_area,
                "red_cov_pre": f"{cov_r_pre:.3f}",
                "blue_cov_pre": f"{cov_b_pre:.3f}",
                "yellow_cov_pre": f"{cov_y_pre:.3f}",
                "best_post_morph_iou": f"{best_iou:.3f}",
                "best_morph_channel": best_c,
                "most_likely_failure_stage": failure_stage,
            })

    audit_csv = out_dir / "missed_signs_audit_records.csv"
    with open(audit_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    print(f"[Step 1 Audit Complete] Saved 29 ground-truth diagnostic records to: {audit_csv}\n")
    return audit_csv


def extract_candidates_phase_g(
    cl_mask: np.ndarray,
    config: PipelineConfig,
    aux_masks: Dict[str, np.ndarray],
    image_shape: Tuple[int, int],
    enhanced_bgr: np.ndarray,
    color_channel: str,
    img_name: str,
    gt_boxes_for_image: List[Dict[str, Any]],
    relax_filter11_variance: bool = False,
) -> List[Dict[str, Any]]:
    """Extract candidate bounding boxes applying Phase F Configuration E pole-tolerant shape filters."""
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
        # In Condition C, blue signs have high internal variance (white arrow on deep blue background);
        # avoid false background rejection for blue sign contours
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

        # Apply Phase F Configuration E pole-tolerant shape condition
        is_accepted, _ = evaluate_shape_condition_phase_f(
            shape_feats, condition_mode="E_pole_tolerant", cnt=cnt, crop_mask=crop_mask
        )

        if is_accepted:
            passed_candidates.append({
                "box": (x, y, w, h),
                "channel": color_channel,
            })

    return passed_candidates


def run_phase_g_condition(
    cond_name: str,
    pipeline_cfg: PipelineConfig,
    sample_images: List[str],
    all_gt: Dict[str, List[Dict]],
    gtsdb_dir: Path,
    out_dir: Path,
    relax_filter11_var: bool = False,
) -> Dict[str, Any]:
    """Execute a single experimental condition for Phase G."""
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
            c_cands = extract_candidates_phase_g(
                cl_c_mask, pipeline_cfg, c_aux, img_bgr.shape[:2], enh,
                color_channel=c_name, img_name=img_name,
                gt_boxes_for_image=gt_boxes,
                relax_filter11_variance=relax_filter11_var,
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
    }


def main():
    root = Path(__file__).resolve().parent
    ds_cfg = DatasetConfig()
    gtsdb_dir = root / ds_cfg.gtsdb_dir
    gt_file = root / ds_cfg.gtsdb_gt_file
    out_dir = root / "experiments_phase_g"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_images = [f"{i:05d}.ppm" for i in range(20)]
    all_gt, _ = load_gtsdb_annotations(gt_file, gtsdb_dir)

    # Base configuration: Phase F Configuration E (Pole-Tolerant Geometry)
    cfg_base = PipelineConfig()
    cfg_base.stage2.independent_color_morphology = True
    cfg_base.stage2.morph_kernel_size = 3
    cfg_base.stage2.morph_close_iterations = 3
    cfg_base.stage2.morph_open_iterations = 1
    cfg_base.stage2.filter9_mode = "alternative"
    cfg_base.stage2.filter11_max_area = 8000
    cfg_base.stage2.nms_iou_threshold = 0.40

    # Step 1: Audit missed signs
    audit_missed_signs(sample_images, all_gt, gtsdb_dir, out_dir, cfg_base)

    # Step 2: Define controlled experiment configurations
    experiments = []

    # A. Unmodified Phase F Configuration E baseline
    experiments.append({
        "name": "A_Baseline_PhaseF_ConfigE",
        "cfg": cfg_base,
        "relax_f11_var": False,
        "description": "Unmodified Phase F Config E baseline (hsv_red: S>=50, lab_red: a*>=140, hsv_blue: S>=40, lab_blue: b*<=115, hsv_yellow: S>=50, lab_yellow: b*>=140)"
    })

    # B. Change red segmentation thresholds only
    # S_min: 50 -> 35, a*_min: 140 -> 128 (captures desaturated/shadowed signs like 00005, 00006, 00017)
    cfg_b = PipelineConfig()
    cfg_b.stage2 = Stage2Config(**cfg_base.stage2.__dict__)
    cfg_b.stage2.hsv_red_lower1 = (0, 35, 40)
    cfg_b.stage2.hsv_red_lower2 = (155, 35, 40)
    cfg_b.stage2.lab_red_lower = (20, 128, 100)
    experiments.append({
        "name": "B_Red_Thresholds_Relaxed",
        "cfg": cfg_b,
        "relax_f11_var": False,
        "description": "Red thresholds relaxed only: hsv_red S>=35 (was 50), lab_red a*>=128 (was 140)"
    })

    # C. Change blue segmentation thresholds only
    # S_min: 40 -> 30, H_upper: 135 -> 140, b*_upper: 115 -> 125, plus relax F11 variance for blue
    cfg_c = PipelineConfig()
    cfg_c.stage2 = Stage2Config(**cfg_base.stage2.__dict__)
    cfg_c.stage2.hsv_blue_lower = (95, 30, 40)
    cfg_c.stage2.hsv_blue_upper = (140, 255, 255)
    cfg_c.stage2.lab_blue_upper = (230, 145, 125)
    experiments.append({
        "name": "C_Blue_Thresholds_Relaxed",
        "cfg": cfg_c,
        "relax_f11_var": True,
        "description": "Blue thresholds relaxed: hsv_blue S>=30 (was 40), H<=140 (was 135), lab_blue b*<=125 (was 115), F11 variance adjusted for blue signs"
    })

    # D. Change yellow segmentation thresholds only
    # S_min: 50 -> 35, b*_min: 140 -> 130
    cfg_d = PipelineConfig()
    cfg_d.stage2 = Stage2Config(**cfg_base.stage2.__dict__)
    cfg_d.stage2.hsv_yellow_lower = (14, 35, 40)
    cfg_d.stage2.lab_yellow_lower = (40, 100, 130)
    experiments.append({
        "name": "D_Yellow_Thresholds_Relaxed",
        "cfg": cfg_d,
        "relax_f11_var": False,
        "description": "Yellow thresholds relaxed: hsv_yellow S>=35 (was 50), lab_yellow b*>=130 (was 140)"
    })

    # E. Combine only the best-supported changes
    cfg_e = PipelineConfig()
    cfg_e.stage2 = Stage2Config(**cfg_base.stage2.__dict__)
    # Combine Red (B) and Blue (C)
    cfg_e.stage2.hsv_red_lower1 = (0, 35, 40)
    cfg_e.stage2.hsv_red_lower2 = (155, 35, 40)
    cfg_e.stage2.lab_red_lower = (20, 128, 100)
    cfg_e.stage2.hsv_blue_lower = (95, 30, 40)
    cfg_e.stage2.hsv_blue_upper = (140, 255, 255)
    cfg_e.stage2.lab_blue_upper = (230, 145, 125)
    experiments.append({
        "name": "E_Combined_Best_Supported_Colors",
        "cfg": cfg_e,
        "relax_f11_var": True,
        "description": "Combined best-supported thresholds: relaxed Red (S>=35, a*>=128) + relaxed Blue (S>=30, b*<=125, F11 var adjustment)"
    })

    summaries = []
    base_tp = None
    base_fp = None

    for exp in experiments:
        res = run_phase_g_condition(
            cond_name=exp["name"],
            pipeline_cfg=exp["cfg"],
            sample_images=sample_images,
            all_gt=all_gt,
            gtsdb_dir=gtsdb_dir,
            out_dir=out_dir,
            relax_filter11_var=exp["relax_f11_var"],
        )
        res["description"] = exp["description"]
        if exp["name"] == "A_Baseline_PhaseF_ConfigE":
            base_tp = res["tp"]
            base_fp = res["fp"]
            res["tp_change_vs_base"] = 0
            res["fp_change_vs_base"] = 0
        else:
            res["tp_change_vs_base"] = res["tp"] - base_tp
            res["fp_change_vs_base"] = res["fp"] - base_fp
        summaries.append(res)

    summary_path = out_dir / "phase_g_comparison_summary.csv"
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
    print("PHASE G COLOR SEGMENTATION EXPERIMENT SUITE SUMMARY (One-to-One IoU >= 0.50)")
    print("=" * 115)
    print(f"{'Experiment':<35} | {'PreNMS':<6} | {'PostNMS':<7} | {'TP':<3} | {'FP':<6} | {'FN':<3} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'TP +/-':<6} | {'FP +/-':<7}")
    print("-" * 115)
    for r in summaries:
        print(f"{r['experiment']:<35} | {r['pre_nms_cands']:<6d} | {r['post_nms_cands']:<7d} | {r['tp']:<3d} | {r['fp']:<6d} | {r['fn']:<3d} | {r['precision']:<7.4f} | {r['recall']:<7.4f} | {r['f1']:<7.4f} | {r['tp_change_vs_base']:<+6d} | {r['fp_change_vs_base']:<+7d}")
    print("=" * 115)
    print(f"Master comparison saved to: {summary_path}\n")


if __name__ == "__main__":
    main()
