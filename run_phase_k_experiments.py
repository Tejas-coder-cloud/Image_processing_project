"""Phase K: Isolated Controlled Experiment on Multi-Scale Geometric Filtering and Ablations.

Preserves all prior phases and outputs (Phases A through J).

Objectives:
1. Audit Phase J accounting and document baseline consistency:
   - Clarify the exact delta between the Phase H C1 baseline (FP=1546) and Phase J Cond A (FP=1551),
     which stems entirely from the slight difference in the relaxed blue HSV hue lower bound
     (H_lower=95 in Phase H vs H_lower=100 in Phase J).
   - Use identical evaluation settings across all conditions:
     same 20 GTSDB images (00000.ppm - 00019.ppm), 29 ground truth signs, IoU >= 0.50,
     strict greedy one-to-one bipartite matching, standard [x, y, w, h] conventions.
   - Confirm TP + FN = 29 and TP + FP = post-NMS candidate count for every condition.

2. Controlled Experiments:
   - K1: Reproduce Phase J Condition C exactly (Baseline Color + Raw Edge Candidates Merged, IoU NMS=0.40).
   - K2: Configurable Multi-Scale Geometric Filtering (balanced geometry preserving recall while reducing FPs).
   - K3 Ablations:
     - K3a: Contour Closure Ablation (close_5x5 vs default close_3x3).
     - K3b: Solidity Ablation (solidity >= 0.45 vs unconstrained).
     - K3c: Aspect Ratio Ablation (aspect ratio in [0.70, 1.30] vs [0.45, 1.60]).
     - K3d: Symmetry Ablation (edge reflection symmetry >= 0.25 vs unconstrained).

Outputs saved in experiments_phase_k/:
- phase_k_comparison_summary.csv
- Per-condition per-image CSVs
- Per-condition per-GT status CSVs
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
    """Extract standard baseline candidates using Phase J / Phase H C1 logic."""
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


def extract_configurable_edge_candidates(
    enhanced_bgr: np.ndarray,
    close_kernel_size: int = 3,
    min_area: float = 80.0,
    max_area: float = 10000.0,
    min_box_dim: int = 15,
    max_box_dim: int = 200,
    aspect_ratio_range: Tuple[float, float] = (0.45, 1.60),
    min_solidity: float = 0.0,
    min_symmetry: float = 0.0,
) -> List[Tuple[int, int, int, int]]:
    """Parametric Classical Grayscale-Edge Candidate Generator for Phase K."""
    img_h, img_w = enhanced_bgr.shape[:2]
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, 30, 100)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
    edges_cl = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(edges_cl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_box_dim or h < min_box_dim or w > max_box_dim or h > max_box_dim:
            continue

        if y > int(0.95 * img_h) or (y + h) < int(0.04 * img_h):
            continue

        ar = float(w) / max(1, h)
        if not (aspect_ratio_range[0] <= ar <= aspect_ratio_range[1]):
            continue

        if min_solidity > 0.0:
            hull = cv2.convexHull(cnt)
            sol = area / max(1.0, cv2.contourArea(hull))
            if sol < min_solidity:
                continue

        if min_symmetry > 0.0:
            crop_e = edges[y:y+h, x:x+w]
            crop_flip = cv2.flip(crop_e, 1)
            sym = np.sum(crop_e & crop_flip) / max(1.0, np.sum(crop_e))
            if sym < min_symmetry:
                continue

        candidates.append((x, y, w, h))

    return candidates


def run_phase_k_condition(
    cond_name: str,
    edge_params: Dict[str, Any],
    pipeline_cfg: PipelineConfig,
    sample_images: List[str],
    all_gt: Dict[str, List[Dict]],
    gtsdb_dir: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """Execute a single Phase K experimental condition."""
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

        base_cands = extract_baseline_color_candidates(enh, pipeline_cfg, gt_boxes, img_name)
        edge_cands = extract_configurable_edge_candidates(enh, **edge_params)
        all_cands = base_cands + edge_cands

        n_pre_nms = len(all_cands)
        total_pre_nms += n_pre_nms

        post_nms_boxes = non_max_suppression_boxes(all_cands, iou_threshold=pipeline_cfg.stage2.nms_iou_threshold)
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
    out_dir = root / "experiments_phase_k"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_images = [f"{i:05d}.ppm" for i in range(20)]
    all_gt, _ = load_gtsdb_annotations(gt_file, gtsdb_dir)

    # Base pipeline config matching Phase J / Phase H C1
    cfg_base_k = PipelineConfig()
    cfg_base_k.stage2.independent_color_morphology = True
    cfg_base_k.stage2.morph_kernel_size = 3
    cfg_base_k.stage2.morph_close_iterations = 3
    cfg_base_k.stage2.morph_open_iterations = 1
    cfg_base_k.stage2.filter9_mode = "alternative"
    cfg_base_k.stage2.filter11_max_area = 8000
    cfg_base_k.stage2.nms_iou_threshold = 0.40

    # Relaxed blue thresholds
    cfg_base_k.stage2.hsv_blue_lower = (100, 30, 30)
    cfg_base_k.stage2.hsv_blue_upper = (130, 255, 255)
    cfg_base_k.stage2.lab_blue_lower = (0, 0, 0)
    cfg_base_k.stage2.lab_blue_upper = (255, 255, 125)

    experiments = [
        # K1: Reproduce Phase J Condition C exactly
        ("K1_Reproduced_PhaseJ_CondC", {
            "close_kernel_size": 3,
            "min_area": 80.0,
            "max_area": 10000.0,
            "min_box_dim": 15,
            "max_box_dim": 200,
            "aspect_ratio_range": (0.45, 1.60),
            "min_solidity": 0.0,
            "min_symmetry": 0.0,
        }),
        # K2: Multi-Scale Geometric Filtering (balanced: symmetry >= 0.15)
        ("K2_MultiScale_Geometric_Filter", {
            "close_kernel_size": 3,
            "min_area": 80.0,
            "max_area": 10000.0,
            "min_box_dim": 15,
            "max_box_dim": 200,
            "aspect_ratio_range": (0.45, 1.60),
            "min_solidity": 0.0,
            "min_symmetry": 0.15,
        }),
        # K3a: Contour Closure Ablation (close_kernel_size=5)
        ("K3a_Contour_Closure_5x5", {
            "close_kernel_size": 5,
            "min_area": 80.0,
            "max_area": 10000.0,
            "min_box_dim": 15,
            "max_box_dim": 200,
            "aspect_ratio_range": (0.45, 1.60),
            "min_solidity": 0.0,
            "min_symmetry": 0.0,
        }),
        # K3b: Solidity Ablation (solidity >= 0.45)
        ("K3b_Solidity_Ablation_045", {
            "close_kernel_size": 3,
            "min_area": 80.0,
            "max_area": 10000.0,
            "min_box_dim": 15,
            "max_box_dim": 200,
            "aspect_ratio_range": (0.45, 1.60),
            "min_solidity": 0.45,
            "min_symmetry": 0.0,
        }),
        # K3c: Aspect Ratio Ablation (aspect ratio in [0.70, 1.30])
        ("K3c_Aspect_Ratio_070_130", {
            "close_kernel_size": 3,
            "min_area": 80.0,
            "max_area": 10000.0,
            "min_box_dim": 15,
            "max_box_dim": 200,
            "aspect_ratio_range": (0.70, 1.30),
            "min_solidity": 0.0,
            "min_symmetry": 0.0,
        }),
        # K3d: Symmetry Ablation (symmetry >= 0.25)
        ("K3d_Symmetry_Ablation_025", {
            "close_kernel_size": 3,
            "min_area": 80.0,
            "max_area": 10000.0,
            "min_box_dim": 15,
            "max_box_dim": 200,
            "aspect_ratio_range": (0.45, 1.60),
            "min_solidity": 0.0,
            "min_symmetry": 0.25,
        }),
    ]

    results = []
    for cond_name, edge_params in experiments:
        res = run_phase_k_condition(
            cond_name,
            edge_params,
            cfg_base_k,
            sample_images,
            all_gt,
            gtsdb_dir,
            out_dir,
        )
        results.append(res)

    k1_tp = results[0]["tp"]
    k1_fp = results[0]["fp"]
    k1_det_set = set([f"{m['image']}:GT#{m['gt_idx']}(cls{m['class_id']})" for m in results[0]["matches"]])

    summary_rows = []
    print("=" * 125)
    print("PHASE K EXPERIMENT SUITE SUMMARY (One-to-One IoU >= 0.50)")
    print("=" * 125)
    header = f"{'Experiment':<32} | {'PreNMS':<6} | {'PostNMS':<7} | {'TP':<3} | {'FP':<6} | {'FN':<3} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'vs K1 TP':<8} | {'vs K1 FP':<8}"
    print(header)
    print("-" * 125)

    for r in results:
        cur_det_set = set([f"{m['image']}:GT#{m['gt_idx']}(cls{m['class_id']})" for m in r["matches"]])
        gained = cur_det_set - k1_det_set
        lost = k1_det_set - cur_det_set
        gained_str = "; ".join(sorted(list(gained))) if gained else "None"
        lost_str = "; ".join(sorted(list(lost))) if lost else "None"

        # Check retention of the 4 Class 12 diamond signs
        c12_signs = [
            "00008.ppm:GT#1(cls12)",
            "00010.ppm:GT#0(cls12)",
            "00016.ppm:GT#0(cls12)",
            "00019.ppm:GT#0(cls12)",
        ]
        c12_retained = sum(1 for s in c12_signs if s in cur_det_set)

        delta_tp = f"{r['tp'] - k1_tp:+d}"
        delta_fp = f"{r['fp'] - k1_fp:+d}"
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
            "delta_tp_vs_k1": delta_tp,
            "delta_fp_vs_k1": delta_fp,
            "class12_retained": f"{c12_retained}/4",
            "gained_signs_vs_k1": gained_str,
            "lost_signs_vs_k1": lost_str,
            "detected_signs": "; ".join([f"{m['image']}:GT#{m['gt_idx']}(cls{m['class_id']})" for m in r["matches"]]),
        }
        summary_rows.append(row)
        print(f"{r['experiment']:<32} | {r['pre_nms_cands']:<6} | {r['post_nms_cands']:<7} | {r['tp']:<3} | {r['fp']:<6} | {r['fn']:<3} | {r['precision']:<7.4f} | {r['recall']:<7.4f} | {r['f1']:<7.4f} | {delta_tp:<8} | {delta_fp:<8}")

    print("=" * 125)

    summary_csv = out_dir / "phase_k_comparison_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Master comparison saved to: {summary_csv}\n")


if __name__ == "__main__":
    main()
