"""Phase J: Controlled Grayscale-Edge and Geometric Candidate-Generation Experiment.

Preserves all prior phases and outputs (Phases A through I).

Objective:
Test whether a parallel classical candidate generator using grayscale preprocessing,
Canny edge detection, and contour/line/diamond geometry can propose candidate boxes
capable of detecting signs (including yellow diamond Class 12 signs) without requiring
yellow color coverage, and evaluate its impact when merged with the Phase H C1 baseline.

Baseline:
- Phase H C1, reproduced in Phase I Condition A:
  Relaxed blue segmentation, blue Filter 11 variance adjustment, yellow morphology
  with close=0 and open=0, pole-tolerant geometry on red/blue/yellow masks.
- 20 GTSDB development images (00000.ppm - 00019.ppm), 29 ground truth signs.
- Evaluation: IoU >= 0.50, strict one-to-one bipartite matching.

Configurations Evaluated:
A. Phase H C1 Baseline Unchanged.
B. Grayscale-Edge Candidates Alone (Raw Edge Contour Candidates).
C. Baseline Candidates Merged with Edge Candidates (Union + NMS).
D. Geometric Diamond/Symmetry Filtered Edge Candidates Merged with Baseline.

Outputs stored in experiments_phase_j/:
- phase_j_comparison_summary.csv
- phase_j_edge_diagnostics.csv
- Per-configuration per-image CSVs
- Per-configuration per-GT status CSVs
- Annotated overlay images for each condition
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


def extract_baseline_color_candidates(
    enhanced_bgr: np.ndarray,
    config: PipelineConfig,
    gt_boxes_for_image: List[Dict[str, Any]],
    img_name: str,
) -> List[Tuple[int, int, int, int]]:
    """Extract standard baseline candidates using Phase H C1 / Phase I Cond A logic."""
    img_h, img_w = enhanced_bgr.shape[:2]
    image_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    color_masks = segment_traffic_sign_colors(enhanced_bgr, config.stage2)
    cfg = config.stage2

    white_mask = color_masks.get("mask_white")
    red_mask = color_masks.get("mask_red")
    yellow_mask = color_masks.get("mask_yellow")
    blue_mask = color_masks.get("mask_blue")

    raw_cands = []

    for c_name in ["red", "blue", "yellow"]:
        raw_mask = color_masks[f"mask_{c_name}"]
        if c_name == "yellow":
            cl_mask = raw_mask  # close=0, open=0
        else:
            cl_mask = apply_morphology(raw_mask, cfg.morph_kernel_size,
                                       cfg.morph_close_iterations,
                                       cfg.morph_open_iterations)

        contours, _ = cv2.findContours(cl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

            crop_combined = cl_mask[y:y + h, x:x + w]
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
            if c_name == "blue":
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
                raw_cands.append((x, y, w, h))

    return raw_cands


def extract_grayscale_edge_candidates(
    enhanced_bgr: np.ndarray,
    canny_thresh: Tuple[int, int] = (30, 100),
    apply_geometric_filter: bool = False,
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """Parallel Classical Grayscale Edge Candidate Generator (J1 & J3).

    Uses:
    1. Grayscale conversion and Gaussian smoothing (5x5, sigma=1.0)
    2. Canny edge detector (30, 100)
    3. Morphological edge closing (3x3 rect) to seal boundary contours
    4. Contour extraction with hierarchy (RETR_TREE)
    5. Standard bounding box & scale constraints:
       - 15 <= w <= 200, 15 <= h <= 200, area in [80, 10000]
       - Aspect ratio in [0.45, 1.60] (accommodating pole attachments and diamond tilts)
    6. Optional geometric diamond/symmetry filtering (Condition D):
       - Solidity >= 0.60
       - Min area >= 200
       - Horizontal edge symmetry >= 0.20
    """
    img_h, img_w = enhanced_bgr.shape[:2]
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, canny_thresh[0], canny_thresh[1])

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_cl = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(edges_cl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80 or area > 10000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 15 or h < 15 or w > 200 or h > 200:
            continue

        # Spatial road/sky horizon bounds (avoid top 4% and bottom 5%)
        if y > int(0.95 * img_h) or (y + h) < int(0.04 * img_h):
            continue

        ar = float(w) / max(1, h)
        if not (0.45 <= ar <= 1.60):
            continue

        if apply_geometric_filter:
            # Modest diamond/polygon geometric criteria:
            if area < 200:
                continue
            hull = cv2.convexHull(cnt)
            solidity = area / max(1.0, cv2.contourArea(hull))
            if solidity < 0.60:
                continue
            # Horizontal reflection symmetry of internal edge pixels
            crop_e = edges[y:y+h, x:x+w]
            crop_flip = cv2.flip(crop_e, 1)
            sym = np.sum(crop_e & crop_flip) / max(1.0, np.sum(crop_e))
            if sym < 0.20:
                continue

        candidates.append((x, y, w, h))

    return candidates, edges


def run_targeted_j2_diagnostics(
    sample_images: List[str],
    all_gt: Dict[str, List[Dict]],
    gtsdb_dir: Path,
    out_dir: Path,
    cfg_base: PipelineConfig,
) -> Path:
    """J2 Targeted Diagnostics on the four Class 12 Yellow Diamond Signs."""
    targets = [("00008.ppm", 1), ("00010.ppm", 0), ("00016.ppm", 0), ("00019.ppm", 0)]
    records = []

    for img_name, gi in targets:
        img_path = gtsdb_dir / img_name
        img_bgr = cv2.imread(str(img_path))
        s1 = run_stage1(img_bgr, cfg_base.stage1)
        enh = s1["enhanced_image"]
        gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)

        gt = all_gt[img_name][gi]
        gx, gy, gw, gh = gt["x"], gt["y"], gt["w"], gt["h"]
        gt_area = gw * gh

        # Raw edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        edges = cv2.Canny(blurred, 30, 100)
        gt_edge_crop = edges[gy:gy+gh, gx:gx+gw]
        edge_cov = float(np.count_nonzero(gt_edge_crop)) / gt_area

        # Raw edge contours
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_cl = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(edges_cl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        best_iou_raw = 0.0
        best_box_raw = None
        best_cnt_raw = None
        for cnt in cnts:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            iou = compute_iou_xywh((gx, gy, gw, gh), (bx, by, bw, bh))
            if iou > best_iou_raw:
                best_iou_raw = iou
                best_box_raw = (bx, by, bw, bh)
                best_cnt_raw = cnt

        # Filtered edge contours (solidity >= 0.60, area >= 200, ar in [0.5, 1.5])
        best_iou_geom = 0.0
        best_box_geom = None
        rejection_reason = "RETAINED"
        if best_cnt_raw is not None:
            area_cnt = cv2.contourArea(best_cnt_raw)
            hull = cv2.convexHull(best_cnt_raw)
            solidity = area_cnt / max(1.0, cv2.contourArea(hull))
            bx, by, bw, bh = best_box_raw
            ar = float(bw) / max(1, bh)
            crop_e = edges[by:by+bh, bx:bx+bw]
            sym = np.sum(crop_e & cv2.flip(crop_e, 1)) / max(1.0, np.sum(crop_e))

            if area_cnt < 200:
                rejection_reason = "AREA_BELOW_200"
            elif solidity < 0.60:
                rejection_reason = "SOLIDITY_BELOW_0.60"
            elif not (0.45 <= ar <= 1.60):
                rejection_reason = "ASPECT_RATIO_OUT_OF_BOUNDS"
            elif sym < 0.20:
                rejection_reason = "SYMMETRY_BELOW_0.20"
            else:
                best_iou_geom = best_iou_raw
                best_box_geom = best_box_raw

        # Measure background edge density in a typical non-sign patch (e.g., foliage/sky)
        bg_patch = edges[50:150, 50:150]
        bg_edge_density = float(np.count_nonzero(bg_patch)) / (100 * 100)

        records.append({
            "image": img_name,
            "gt_idx": gi,
            "class_id": gt["class_id"],
            "gt_box": f"({gx},{gy},{gw},{gh})",
            "gt_edge_coverage": f"{edge_cov:.4f}",
            "bg_edge_density_sample": f"{bg_edge_density:.4f}",
            "best_raw_edge_box": str(best_box_raw),
            "best_raw_edge_iou": f"{best_iou_raw:.4f}",
            "best_geom_edge_box": str(best_box_geom),
            "best_geom_edge_iou": f"{best_iou_geom:.4f}",
            "meets_iou_050_raw": (best_iou_raw >= 0.50),
            "meets_iou_050_geom": (best_iou_geom >= 0.50),
            "geom_filter_status": rejection_reason,
        })

    out_csv = out_dir / "phase_j_edge_diagnostics.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    print(f"[J2 Diagnostic Complete] Saved edge diagnostic records to: {out_csv}\n")
    return out_csv


def run_phase_j_condition(
    cond_name: str,
    cond_type: str,  # 'A_baseline', 'B_edge_alone', 'C_baseline_plus_edge', 'D_baseline_plus_geom_edge'
    pipeline_cfg: PipelineConfig,
    sample_images: List[str],
    all_gt: Dict[str, List[Dict]],
    gtsdb_dir: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """Execute a single Phase J experimental condition."""
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

        candidates = []

        if cond_type == "A_baseline":
            candidates = extract_baseline_color_candidates(enh, pipeline_cfg, gt_boxes, img_name)
        elif cond_type == "B_edge_alone":
            edge_cands, _ = extract_grayscale_edge_candidates(enh, apply_geometric_filter=False)
            candidates = edge_cands
        elif cond_type == "C_baseline_plus_edge":
            base_cands = extract_baseline_color_candidates(enh, pipeline_cfg, gt_boxes, img_name)
            edge_cands, _ = extract_grayscale_edge_candidates(enh, apply_geometric_filter=False)
            candidates = base_cands + edge_cands
        elif cond_type == "D_baseline_plus_geom_edge":
            base_cands = extract_baseline_color_candidates(enh, pipeline_cfg, gt_boxes, img_name)
            edge_cands, _ = extract_grayscale_edge_candidates(enh, apply_geometric_filter=True)
            candidates = base_cands + edge_cands

        n_pre_nms = len(candidates)
        total_pre_nms += n_pre_nms

        # NMS merging (IoU = 0.40)
        post_nms_boxes = non_max_suppression_boxes(candidates, iou_threshold=pipeline_cfg.stage2.nms_iou_threshold)
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
            print(f"  [{idx:2d}/20] {img_name} -> Pre={n_pre_nms:5d} Post={n_post_nms:4d} | TP={eval_res['tp']} | {match_info}")
        else:
            print(f"  [{idx:2d}/20] {img_name} -> Pre={n_pre_nms:5d} Post={n_post_nms:4d} | TP=0")

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
    out_dir = root / "experiments_phase_j"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_images = [f"{i:05d}.ppm" for i in range(20)]
    all_gt, _ = load_gtsdb_annotations(gt_file, gtsdb_dir)

    # Base configuration: Phase H C1
    cfg_base_j = PipelineConfig()
    cfg_base_j.stage2.independent_color_morphology = True
    cfg_base_j.stage2.morph_kernel_size = 3
    cfg_base_j.stage2.morph_close_iterations = 3
    cfg_base_j.stage2.morph_open_iterations = 1
    cfg_base_j.stage2.filter9_mode = "alternative"
    cfg_base_j.stage2.filter11_max_area = 8000
    cfg_base_j.stage2.nms_iou_threshold = 0.40

    # Relaxed blue segmentation thresholds validated in Phase H & I
    cfg_base_j.stage2.hsv_blue_lower = (100, 30, 30)
    cfg_base_j.stage2.hsv_blue_upper = (130, 255, 255)
    cfg_base_j.stage2.lab_blue_lower = (0, 0, 0)
    cfg_base_j.stage2.lab_blue_upper = (255, 255, 125)

    # 1. Run J2 Diagnostics on the four Class 12 Yellow Diamond Signs
    run_targeted_j2_diagnostics(sample_images, all_gt, gtsdb_dir, out_dir, cfg_base_j)

    # 2. Run Controlled Suite: Conditions A, B, C, D
    results = []

    # Condition A: Baseline Phase H C1 unchanged
    res_a = run_phase_j_condition(
        "A_PhaseH_C1_Baseline_Unchanged",
        "A_baseline",
        cfg_base_j,
        sample_images,
        all_gt,
        gtsdb_dir,
        out_dir,
    )
    results.append(res_a)

    # Condition B: Grayscale-Edge Candidates Alone
    res_b = run_phase_j_condition(
        "B_Grayscale_Edge_Candidates_Alone",
        "B_edge_alone",
        cfg_base_j,
        sample_images,
        all_gt,
        gtsdb_dir,
        out_dir,
    )
    results.append(res_b)

    # Condition C: Baseline Candidates Plus Edge Candidates (Union + NMS)
    res_c = run_phase_j_condition(
        "C_Baseline_Plus_Edge_Candidates",
        "C_baseline_plus_edge",
        cfg_base_j,
        sample_images,
        all_gt,
        gtsdb_dir,
        out_dir,
    )
    results.append(res_c)

    # Condition D: Baseline Plus Geometrically Filtered Edge Candidates
    res_d = run_phase_j_condition(
        "D_Baseline_Plus_Geometric_Edge_Candidates",
        "D_baseline_plus_geom_edge",
        cfg_base_j,
        sample_images,
        all_gt,
        gtsdb_dir,
        out_dir,
    )
    results.append(res_d)

    # 3. Master Comparison Summary
    base_tp = res_a["tp"]
    base_fp = res_a["fp"]

    summary_rows = []
    print("=" * 115)
    print("PHASE J EXPERIMENT SUITE SUMMARY (One-to-One IoU >= 0.50)")
    print("=" * 115)
    header = f"{'Experiment':<42} | {'PreNMS':<6} | {'PostNMS':<7} | {'TP':<3} | {'FP':<6} | {'FN':<3} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'TP +/-':<6} | {'FP +/-':<6}"
    print(header)
    print("-" * 115)

    for r in results:
        delta_tp = f"{r['tp'] - base_tp:+d}"
        delta_fp = f"{r['fp'] - base_fp:+d}"
        row = {
            "experiment": r["experiment"],
            "pre_nms": r["pre_nms_cands"],
            "post_nms": r["post_nms_cands"],
            "tp": r["tp"],
            "fp": r["fp"],
            "fn": r["fn"],
            "precision": f"{r['precision']:.4f}",
            "recall": f"{r['recall']:.4f}",
            "f1": f"{r['f1']:.4f}",
            "delta_tp": delta_tp,
            "delta_fp": delta_fp,
            "detected_signs": "; ".join([f"{m['image']}:GT#{m['gt_idx']}(cls{m['class_id']})" for m in r["matches"]]),
        }
        summary_rows.append(row)
        print(f"{r['experiment']:<42} | {r['pre_nms_cands']:<6} | {r['post_nms_cands']:<7} | {r['tp']:<3} | {r['fp']:<6} | {r['fn']:<3} | {r['precision']:<7.4f} | {r['recall']:<7.4f} | {r['f1']:<7.4f} | {delta_tp:<6} | {delta_fp:<6}")

    print("=" * 115)

    summary_csv = out_dir / "phase_j_comparison_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Master comparison saved to: {summary_csv}\n")


if __name__ == "__main__":
    main()
