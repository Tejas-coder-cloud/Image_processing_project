"""Phase L — Experiment A: Secondary Filters Applied to Edge Candidates.

Tests whether the three secondary filters already used by the color branch
can reduce edge-branch false positives when applied to edge candidates.

Conditions:
- A0: Phase K2 baseline unchanged.
- A1: K2 + _is_large_background_region on edge candidates.
- A2: K2 + _is_likely_vehicle_region on edge candidates.
- A3: K2 + _is_traffic_light on edge candidates.
- A4: K2 + all three filters on edge candidates.

Filter dependency analysis (from morphology.py):
- _is_large_background_region(image_gray, x, y, w, h, max_area):
    Requires only grayscale image and box coordinates. Safe to apply to edge cands.
- _is_likely_vehicle_region(x, y, w, h, img_h, img_w, red_mask, mode):
    With mode='alternative', requires only box coords and image dimensions.
    red_mask is used only in 'current' mode. We use mode='alternative' (no mask needed).
- _is_traffic_light(image_gray, x, y, w, h):
    Requires grayscale image and box coords. Safe to apply to edge cands.

All three filters depend only on box geometry and image_gray — both of which
are available for edge candidates without any color-branch-specific features.

Outputs saved in experiments_phase_l/:
- phase_l_expa_comparison_summary.csv
- phase_l_expa_filter_rejection_log.csv
- Per-condition per-image CSVs
- Per-condition per-GT status CSVs
- Annotated overlay images
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


def load_gtsdb_annotations(gt_path):
    """Load GTSDB annotations from gt.txt file."""
    annotations = {}
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 6:
                filename = parts[0]
                left, top, right, bottom = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                class_id = int(parts[5])
                if filename not in annotations:
                    annotations[filename] = []
                annotations[filename].append({
                    'filename': filename,
                    'left': left,
                    'top': top,
                    'right': right,
                    'bottom': bottom,
                    'class_id': class_id
                })
    return annotations


def compute_iou_xywh(box1, box2):
    """Compute IoU between two boxes in (x, y, w, h) format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to (x1, y1, x2, y2)
    box1_x1, box1_y1, box1_x2, box1_y2 = x1, y1, x1 + w1, y1 + h1
    box2_x1, box2_y1, box2_x2, box2_y2 = x2, y2, x2 + w2, y2 + h2
    
    # Compute intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # Compute union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


# ─── Evaluation ───────────────────────────────────────────────────────────────

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
    matched_gt, matched_det, matches = set(), set(), []

    for iou, gi, di in iou_pairs:
        if gi in matched_gt or di in matched_det:
            continue
        matched_gt.add(gi)
        matched_det.add(di)
        matches.append({
            "gt_idx": gi, "det_idx": di, "iou": iou,
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
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "matches": matches,
        "unmatched_gt": [i for i in range(n_gt) if i not in matched_gt],
        "unmatched_gt_indices": [i for i in range(n_gt) if i not in matched_gt],
        "unmatched_det_indices": [i for i in range(n_det) if i not in matched_det],
    }


# ─── Color Candidate Extraction (Phase K2 / Phase H C1) ───────────────────────

def extract_baseline_color_candidates(
    enhanced_bgr: np.ndarray,
    config: PipelineConfig,
) -> List[Tuple[int, int, int, int]]:
    """Extract color candidates using Phase K2 / Phase H C1 logic (12-filter cascade)."""
    img_h, img_w = enhanced_bgr.shape[:2]
    image_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    color_masks = segment_traffic_sign_colors(enhanced_bgr, config.stage2)
    cfg = config.stage2

    white_mask = color_masks.get("mask_white")
    red_mask   = color_masks.get("mask_red")
    yellow_mask = color_masks.get("mask_yellow")
    blue_mask  = color_masks.get("mask_blue")

    raw_cands = []
    for c_name in ["red", "blue", "yellow"]:
        raw_mask = color_masks[f"mask_{c_name}"]
        cl_mask  = raw_mask if c_name == "yellow" else apply_morphology(
            raw_mask, cfg.morph_kernel_size, cfg.morph_close_iterations, cfg.morph_open_iterations)

        contours, _ = cv2.findContours(cl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg.min_area or area > cfg.max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < cfg.min_box_dim or h < cfg.min_box_dim or w > cfg.max_box_dim or h > cfg.max_box_dim:
                continue
            if not (cfg.min_aspect_ratio <= float(w) / max(1, h) <= cfg.max_aspect_ratio):
                continue
            if y > int(cfg.max_road_y_ratio * img_h) or (y + h) < int(cfg.min_sky_y_ratio * img_h):
                continue

            circ_mode  = getattr(cfg, "circularity_mode", "contour")
            circularity = _compute_circularity(cnt, mode=circ_mode)
            if hasattr(cfg, "min_circularity") and circularity < cfg.min_circularity:
                continue
            convexity = _compute_convexity(cnt)
            if hasattr(cfg, "min_convexity") and convexity < cfg.min_convexity:
                continue

            color_density = _compute_color_density(cl_mask[y:y+h, x:x+w], w, h)
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
                crop_g   = image_gray[y:y+h, x:x+w]
                box_area = w * h
                if box_area > f11_max or (box_area > 800 and float(np.var(crop_g)) < 150):
                    continue
            else:
                if _is_large_background_region(image_gray, x, y, w, h, max_area=f11_max):
                    continue

            w_cnt = cv2.countNonZero(white_mask[y:y+h, x:x+w]) if white_mask is not None else 0
            y_cnt = cv2.countNonZero(yellow_mask[y:y+h, x:x+w]) if yellow_mask is not None else 0
            r_cnt = cv2.countNonZero(red_mask[y:y+h, x:x+w])   if red_mask is not None else 0
            b_cnt = cv2.countNonZero(blue_mask[y:y+h, x:x+w])   if blue_mask is not None else 0
            total_px = max(1, w * h)
            sign_frac = float(r_cnt + b_cnt + y_cnt) / total_px
            if not (sign_frac >= 0.08 or (w_cnt >= 5 and (r_cnt >= 3 or b_cnt >= 3))
                    or (convexity >= 0.60 and circularity >= 0.30) or y_cnt >= 5):
                continue

            shape_feats = compute_shape_features(cnt, x, y, w, h)
            is_accepted, _ = evaluate_shape_condition_phase_f(
                shape_feats, condition_mode="E_pole_tolerant",
                cnt=cnt, crop_mask=cl_mask[y:y+h, x:x+w])
            if is_accepted:
                raw_cands.append((x, y, w, h))

    return raw_cands


# ─── Edge Candidate Extraction with Optional Secondary Filters ─────────────────

def extract_edge_candidates_with_filters(
    enhanced_bgr: np.ndarray,
    image_gray: np.ndarray,
    img_h: int,
    img_w: int,
    apply_background_filter: bool = False,
    apply_vehicle_filter: bool = False,
    apply_traffic_light_filter: bool = False,
    filter_max_area: int = 8000,
) -> Tuple[List[Tuple[int, int, int, int]], Dict[str, int]]:
    """
    Generate edge candidates (K2 base parameters) with optional secondary filters.

    Returns:
        candidates: accepted (x, y, w, h) tuples.
        rejection_log: dict mapping filter name -> count of candidates rejected.

    Filter applicability notes:
    - _is_large_background_region: Uses image_gray + box coords only. Fully applicable.
    - _is_likely_vehicle_region with mode='alternative': Uses box coords + img dimensions only.
        No red_mask required. The 'alternative' mode checks area>8000, wide+low, and very-bottom.
        Safe to apply unchanged.
    - _is_traffic_light: Uses image_gray + box coords only. Fully applicable.
    """
    blurred  = cv2.GaussianBlur(image_gray, (5, 5), 1.0)
    edges    = cv2.Canny(blurred, 30, 100)
    k        = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_cl = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(edges_cl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rejection_log = {
        "base_area":        0,
        "base_box_dim":     0,
        "base_spatial":     0,
        "base_aspect":      0,
        "base_symmetry":    0,
        "f_background":     0,
        "f_vehicle":        0,
        "f_traffic_light":  0,
        "accepted":         0,
    }

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80.0 or area > 10000.0:
            rejection_log["base_area"] += 1
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 15 or h < 15 or w > 200 or h > 200:
            rejection_log["base_box_dim"] += 1
            continue
        if y > int(0.95 * img_h) or (y + h) < int(0.04 * img_h):
            rejection_log["base_spatial"] += 1
            continue
        ar = float(w) / max(1, h)
        if not (0.45 <= ar <= 1.60):
            rejection_log["base_aspect"] += 1
            continue
        # K2 symmetry gate: >= 0.15
        crop_e    = edges[y:y+h, x:x+w]
        crop_flip = cv2.flip(crop_e, 1)
        sym = np.sum(crop_e & crop_flip) / max(1.0, np.sum(crop_e))
        if sym < 0.15:
            rejection_log["base_symmetry"] += 1
            continue

        # Optional secondary filters
        if apply_background_filter:
            if _is_large_background_region(image_gray, x, y, w, h, max_area=filter_max_area):
                rejection_log["f_background"] += 1
                continue

        if apply_vehicle_filter:
            # mode='alternative' requires no red_mask; uses only geometry + position
            if _is_likely_vehicle_region(x, y, w, h, img_h, img_w, red_mask=None, mode="alternative"):
                rejection_log["f_vehicle"] += 1
                continue

        if apply_traffic_light_filter:
            if _is_traffic_light(image_gray, x, y, w, h):
                rejection_log["f_traffic_light"] += 1
                continue

        rejection_log["accepted"] += 1
        candidates.append((x, y, w, h))

    return candidates, rejection_log


# ─── Per-condition runner ──────────────────────────────────────────────────────

def run_phase_l_condition(
    cond_name: str,
    apply_background: bool,
    apply_vehicle: bool,
    apply_traffic_light: bool,
    pipeline_cfg: PipelineConfig,
    sample_images: List[str],
    all_gt: Dict[str, List[Dict]],
    gtsdb_dir: Path,
    out_dir: Path,
    filter_max_area: int = 8000,
) -> Dict[str, Any]:
    """Execute one Phase L experimental condition."""
    cond_dir = out_dir / cond_name
    cond_dir.mkdir(parents=True, exist_ok=True)
    (cond_dir / "overlays").mkdir(exist_ok=True)

    per_image_rows = []
    per_gt_rows    = []
    rejection_rows = []

    totals = dict(gt=0, pre_nms=0, post_nms=0, tp=0, fp=0, fn=0,
                  rej_bg=0, rej_veh=0, rej_tl=0)

    print("=" * 80)
    print(f"RUNNING: {cond_name}  [bg={apply_background} veh={apply_vehicle} tl={apply_traffic_light}]")
    print("=" * 80)

    for idx, img_name in enumerate(sample_images, 1):
        img_bgr = cv2.imread(str(gtsdb_dir / img_name))
        if img_bgr is None:
            continue
        gt_boxes = all_gt.get(img_name, [])
        s1       = run_stage1(img_bgr, pipeline_cfg.stage1)
        enh      = s1["enhanced_image"]
        img_h, img_w = enh.shape[:2]
        gray     = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)

        color_cands = extract_baseline_color_candidates(enh, pipeline_cfg)
        edge_cands, rej = extract_edge_candidates_with_filters(
            enh, gray, img_h, img_w,
            apply_background_filter=apply_background,
            apply_vehicle_filter=apply_vehicle,
            apply_traffic_light_filter=apply_traffic_light,
            filter_max_area=filter_max_area,
        )

        all_cands  = color_cands + edge_cands
        n_pre_nms  = len(all_cands)
        post_boxes = non_max_suppression_boxes(all_cands, pipeline_cfg.stage2.nms_iou_threshold)
        n_post_nms = len(post_boxes)

        eval_res = evaluate_one_to_one(post_boxes, gt_boxes, 0.50)

        totals["gt"]       += len(gt_boxes)
        totals["pre_nms"]  += n_pre_nms
        totals["post_nms"] += n_post_nms
        totals["tp"]       += eval_res["tp"]
        totals["fp"]       += eval_res["fp"]
        totals["fn"]       += eval_res["fn"]
        totals["rej_bg"]   += rej["f_background"]
        totals["rej_veh"]  += rej["f_vehicle"]
        totals["rej_tl"]   += rej["f_traffic_light"]

        matched_map = {m["gt_idx"]: m for m in eval_res["matches"]}
        for gi, gt in enumerate(gt_boxes):
            is_tp = gi in matched_map
            m     = matched_map.get(gi)
            per_gt_rows.append({
                "image": img_name, "gt_idx": gi,
                "class_id": gt["class_id"],
                "gt_box": f"({gt['x']},{gt['y']},{gt['w']},{gt['h']})",
                "status": "DETECTED_TP" if is_tp else "MISSED_FN",
                "matched_det_box": str(m["det_box"]) if is_tp else "None",
                "iou": f"{m['iou']:.4f}" if is_tp else "0.0000",
            })

        overlay = draw_overlay(enh, post_boxes, gt_boxes, eval_res,
                               title=f"{cond_name}: {img_name}")
        cv2.imwrite(str(cond_dir / "overlays" / f"{Path(img_name).stem}_overlay.png"), overlay)

        per_image_rows.append({
            "image": img_name,
            "gt_signs": len(gt_boxes),
            "color_cands": len(color_cands),
            "edge_cands": len(edge_cands),
            "pre_nms": n_pre_nms,
            "post_nms": n_post_nms,
            "tp": eval_res["tp"], "fp": eval_res["fp"], "fn": eval_res["fn"],
            "precision": f"{eval_res['precision']:.4f}",
            "recall":    f"{eval_res['recall']:.4f}",
            "f1":        f"{eval_res['f1']:.4f}",
            "rej_background": rej["f_background"],
            "rej_vehicle":    rej["f_vehicle"],
            "rej_traffic_light": rej["f_traffic_light"],
        })

        rejection_rows.append({
            "image": img_name, "condition": cond_name,
            "edge_base_area": rej["base_area"],
            "edge_base_box_dim": rej["base_box_dim"],
            "edge_base_spatial": rej["base_spatial"],
            "edge_base_aspect": rej["base_aspect"],
            "edge_base_symmetry": rej["base_symmetry"],
            "edge_f_background": rej["f_background"],
            "edge_f_vehicle": rej["f_vehicle"],
            "edge_f_traffic_light": rej["f_traffic_light"],
            "edge_accepted": rej["accepted"],
        })

        if eval_res["tp"] > 0:
            info = ", ".join([f"GT#{m['gt_idx']} cls={m['class_id']} (IoU={m['iou']:.2f})"
                              for m in eval_res["matches"]])
            print(f"  [{idx:2d}/20] {img_name} -> Col={len(color_cands):3d} Edge={len(edge_cands):3d} "
                  f"Post={n_post_nms:4d} | TP={eval_res['tp']} | {info}")
        else:
            print(f"  [{idx:2d}/20] {img_name} -> Col={len(color_cands):3d} Edge={len(edge_cands):3d} "
                  f"Post={n_post_nms:4d} | TP=0")

    # write CSVs
    with open(cond_dir / f"{cond_name}_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        w.writeheader(); w.writerows(per_image_rows)

    with open(cond_dir / f"{cond_name}_gt_status.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_gt_rows[0].keys()))
        w.writeheader(); w.writerows(per_gt_rows)

    with open(cond_dir / f"{cond_name}_rejection_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rejection_rows[0].keys()))
        w.writeheader(); w.writerows(rejection_rows)

    prec = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    rec  = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    f1   = 2 * prec * rec / max(1e-8, prec + rec)

    assert totals["tp"] + totals["fn"] == totals["gt"], \
        f"TP+FN={totals['tp']+totals['fn']} != GT={totals['gt']}"
    assert totals["tp"] + totals["fp"] == totals["post_nms"], \
        f"TP+FP={totals['tp']+totals['fp']} != PostNMS={totals['post_nms']}"

    print(f"\n[{cond_name}] GT={totals['gt']} PreNMS={totals['pre_nms']} "
          f"PostNMS={totals['post_nms']} TP={totals['tp']} FP={totals['fp']} FN={totals['fn']} "
          f"P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    print(f"  Edge rejections: background={totals['rej_bg']} vehicle={totals['rej_veh']} "
          f"traffic_light={totals['rej_tl']}\n")

    return {
        "condition": cond_name,
        "pre_nms": totals["pre_nms"], "post_nms": totals["post_nms"],
        "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"],
        "precision": prec, "recall": rec, "f1": f1,
        "rej_bg": totals["rej_bg"], "rej_veh": totals["rej_veh"], "rej_tl": totals["rej_tl"],
        "matches": [r for r in per_gt_rows if r["status"] == "DETECTED_TP"],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    root     = Path(__file__).resolve().parent
    ds_cfg   = DatasetConfig()
    gtsdb_dir = root / ds_cfg.gtsdb_dir
    gt_file  = root / ds_cfg.gtsdb_gt_file
    out_dir  = root / "experiments_phase_l"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_images = [f"{i:05d}.ppm" for i in range(20)]
    all_gt, _     = load_gtsdb_annotations(gt_file, gtsdb_dir)

    # Phase K2 configuration
    cfg = PipelineConfig()
    cfg.stage2.independent_color_morphology = True
    cfg.stage2.morph_kernel_size            = 3
    cfg.stage2.morph_close_iterations       = 3
    cfg.stage2.morph_open_iterations        = 1
    cfg.stage2.filter9_mode                 = "alternative"
    cfg.stage2.filter11_max_area            = 8000
    cfg.stage2.nms_iou_threshold            = 0.40
    cfg.stage2.hsv_blue_lower               = (100, 30, 30)
    cfg.stage2.hsv_blue_upper               = (130, 255, 255)
    cfg.stage2.lab_blue_lower               = (0, 0, 0)
    cfg.stage2.lab_blue_upper               = (255, 255, 125)

    conditions = [
        # (name,  bg,    veh,   tl)
        ("A0_K2_Baseline",               False, False, False),
        ("A1_Background_Filter",         True,  False, False),
        ("A2_Vehicle_Filter",            False, True,  False),
        ("A3_Traffic_Light_Filter",      False, False, True),
        ("A4_All_Three_Filters",         True,  True,  True),
    ]

    results = []
    for cond_name, bg, veh, tl in conditions:
        res = run_phase_l_condition(
            cond_name, bg, veh, tl, cfg,
            sample_images, all_gt, gtsdb_dir, out_dir,
            filter_max_area=cfg.stage2.filter11_max_area,
        )
        results.append(res)

    # ─── Summary Table ─────────────────────────────────────────────────────────
    a0_tp = results[0]["tp"]
    a0_fp = results[0]["fp"]
    a0_det = {f"{m['image']}:GT#{m['gt_idx']}(cls{m['class_id']})" for m in results[0]["matches"]}
    CLASS12 = {"00008.ppm:GT#1(cls12)", "00010.ppm:GT#0(cls12)",
               "00016.ppm:GT#0(cls12)", "00019.ppm:GT#0(cls12)"}
    CLS21_COLOR = "00014.ppm:GT#0(cls21)"  # detected by color branch only

    summary_rows = []
    print("\n" + "=" * 120)
    print("PHASE L EXPERIMENT A — SUMMARY  (IoU >= 0.50, one-to-one matching)")
    print("=" * 120)
    hdr = (f"{'Condition':<30} | {'PreNMS':>6} | {'Post':>5} | {'TP':>2} | {'FP':>5} | "
           f"{'FN':>2} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | "
           f"{'dTP':>4} | {'dFP':>5} | {'C12':>4} | {'Rej-BG':>6} | {'Rej-V':>5} | {'Rej-TL':>6}")
    print(hdr)
    print("-" * 120)

    for r in results:
        cur_det  = {f"{m['image']}:GT#{m['gt_idx']}(cls{m['class_id']})" for m in r["matches"]}
        gained   = cur_det - a0_det
        lost     = a0_det - cur_det
        c12_kept = len(CLASS12 & cur_det)
        cls21_ok = CLS21_COLOR in cur_det

        row = {
            "condition": r["condition"],
            "pre_nms": r["pre_nms"], "post_nms": r["post_nms"],
            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
            "precision": f"{r['precision']:.4f}",
            "recall":    f"{r['recall']:.4f}",
            "f1":        f"{r['f1']:.4f}",
            "delta_tp_vs_a0": f"{r['tp'] - a0_tp:+d}",
            "delta_fp_vs_a0": f"{r['fp'] - a0_fp:+d}",
            "class12_retained": f"{c12_kept}/4",
            "cls21_00014_color_ok": cls21_ok,
            "rej_background": r["rej_bg"],
            "rej_vehicle":    r["rej_veh"],
            "rej_traffic_light": r["rej_tl"],
            "gained_vs_a0": "; ".join(sorted(gained)) if gained else "None",
            "lost_vs_a0":   "; ".join(sorted(lost))   if lost   else "None",
            "detected_signs": "; ".join(sorted(cur_det)),
        }
        summary_rows.append(row)

        dtp = f"{r['tp'] - a0_tp:+d}"
        dfp = f"{r['fp'] - a0_fp:+d}"
        print(f"{r['condition']:<30} | {r['pre_nms']:>6} | {r['post_nms']:>5} | "
              f"{r['tp']:>2} | {r['fp']:>5} | {r['fn']:>2} | "
              f"{r['precision']:>6.4f} | {r['recall']:>6.4f} | {r['f1']:>6.4f} | "
              f"{dtp:>4} | {dfp:>5} | {c12_kept:>2}/4 | "
              f"{r['rej_bg']:>6} | {r['rej_veh']:>5} | {r['rej_tl']:>6}")
        if lost:
            print(f"  ** LOST vs A0: {'; '.join(sorted(lost))}")
        if gained:
            print(f"  ++ GAINED vs A0: {'; '.join(sorted(gained))}")
    print("=" * 120)

    summary_csv = out_dir / "phase_l_expa_comparison_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"\nSummary CSV saved to: {summary_csv}")


if __name__ == "__main__":
    main()
