"""Phase L  Experiment C: Classical Candidate Confidence Scoring.

Reference: K2 + A3 + B2 (edge min_area=300, traffic-light filter on).
Expected baseline: TP=17, FP=2820, FN=12, post-NMS=2837.

Design:
  1. Reproduce the reference.
  2. Collect per-candidate features (both branches) after NMS.
  3. Analyse feature distributions for GT-matching vs non-matching candidates.
  4. Define a transparent score, sweep thresholds.
  5. Report.

Scoring is applied POST-NMS: the NMS-surviving candidate list is unchanged,
but each surviving box receives a score.  Candidates below a threshold
are removed.  This avoids altering NMS behaviour or candidate ordering.
"""

import csv, math, os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.config import PipelineConfig, DatasetConfig
from src.pipeline import run_stage1
from src.stage2_segmentation.thresholding import segment_traffic_sign_colors
from src.stage2_segmentation.morphology import (
    apply_morphology, _is_traffic_light, non_max_suppression_boxes
)
from evaluate_and_compare import load_gtsdb_annotations, compute_iou_xywh, draw_overlay
from run_phase_l_expa import extract_baseline_color_candidates, evaluate_one_to_one


#  Edge candidate extraction (B2 settings: min_area=300, A3 TL filter)

def extract_edge_candidates_b2(
    enhanced_bgr: np.ndarray,
    image_gray: np.ndarray,
    img_h: int, img_w: int,
) -> List[Tuple[int, int, int, int]]:
    """K2 + A3 + B2 edge candidates."""
    blurred  = cv2.GaussianBlur(image_gray, (5, 5), 1.0)
    edges    = cv2.Canny(blurred, 30, 100)
    k        = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_cl = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(edges_cl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300.0 or area > 10000.0:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 15 or h < 15 or w > 200 or h > 200:
            continue
        if y > int(0.95 * img_h) or (y + h) < int(0.04 * img_h):
            continue
        ar = float(w) / max(1, h)
        if not (0.45 <= ar <= 1.60):
            continue
        crop_e    = edges[y:y+h, x:x+w]
        crop_flip = cv2.flip(crop_e, 1)
        sym = np.sum(crop_e & crop_flip) / max(1.0, np.sum(crop_e))
        if sym < 0.15:
            continue
        if _is_traffic_light(image_gray, x, y, w, h):
            continue
        candidates.append((x, y, w, h))
    return candidates


#  Feature computation for a single candidate box

def compute_candidate_features(
    x: int, y: int, w: int, h: int,
    image_gray: np.ndarray,
    edges_raw: np.ndarray,
    red_mask: np.ndarray,
    blue_mask: np.ndarray,
    yellow_mask: np.ndarray,
    source: str,
    img_h: int, img_w: int,
    other_branch_boxes: List[Tuple[int, int, int, int]],
) -> Dict[str, float]:
    """Compute interpretable features for a single candidate box.

    All features are available for both color and edge candidates.
    Color mask coverage will naturally be low for edge-only FPs and high for
    colour-branch TPs, which is part of the signal.
    """
    box_area = float(w * h)
    ar = float(w) / max(1, h)
    ar_deviation = abs(ar - 1.0)   # 0 = perfectly square

    crop_gray = image_gray[y:y+h, x:x+w]
    gray_var  = float(np.var(crop_gray)) if crop_gray.size > 0 else 0.0
    gray_mean = float(np.mean(crop_gray)) if crop_gray.size > 0 else 0.0

    # Edge density within box
    crop_edge = edges_raw[y:y+h, x:x+w]
    edge_density = float(np.count_nonzero(crop_edge)) / max(1.0, box_area)

    # Edge symmetry within box
    if crop_edge.size > 0 and np.count_nonzero(crop_edge) > 0:
        crop_flip = cv2.flip(crop_edge, 1)
        edge_sym = float(np.sum(crop_edge & crop_flip)) / max(1.0, float(np.sum(crop_edge)))
    else:
        edge_sym = 0.0

    # Colour mask coverage
    r_px = float(np.count_nonzero(red_mask[y:y+h, x:x+w]))
    b_px = float(np.count_nonzero(blue_mask[y:y+h, x:x+w]))
    y_px = float(np.count_nonzero(yellow_mask[y:y+h, x:x+w]))
    color_coverage = (r_px + b_px + y_px) / max(1.0, box_area)

    # Dominant colour fraction (the max of any single colour)
    max_color_px = max(r_px, b_px, y_px)
    dominant_color_frac = max_color_px / max(1.0, box_area)

    # Vertical position
    center_y_ratio = (y + h / 2.0) / max(1, img_h)

    # Cross-branch agreement: highest IoU with any box from the other branch
    best_cross_iou = 0.0
    for ob in other_branch_boxes:
        iou = compute_iou_xywh((x, y, w, h), ob)
        if iou > best_cross_iou:
            best_cross_iou = iou

    return {
        "x": x, "y": y, "w": w, "h": h,
        "box_area": box_area,
        "aspect_ratio": ar,
        "ar_deviation": ar_deviation,
        "gray_variance": gray_var,
        "gray_mean": gray_mean,
        "edge_density": edge_density,
        "edge_symmetry": edge_sym,
        "color_coverage": color_coverage,
        "dominant_color_frac": dominant_color_frac,
        "center_y_ratio": center_y_ratio,
        "cross_branch_iou": best_cross_iou,
        "source": source,
    }


#  Scoring functions

def score_c1_color_presence(feats: Dict[str, float]) -> float:
    """C1  Single-feature baseline: color mask coverage.

    Hypothesis: true traffic signs contain red, blue, or yellow pixels
    while background edge contours do not.  This is the strongest expected
    discriminator because color presence is the fundamental reason we run
    the color branch.
    """
    return feats["color_coverage"]


def score_c2_weighted(feats: Dict[str, float]) -> float:
    """C2  Transparent weighted score using multiple interpretable features.

    Score = w1 * color_coverage
          + w2 * edge_symmetry
          + w3 * cross_branch_agreement
          + w4 * (1 - ar_deviation)   (squareness bonus)

    Weights chosen from feature analysis (see report):
    - color_coverage is the strongest discriminator (w1=0.40)
    - edge_symmetry separates structured from random contours (w2=0.20)
    - cross_branch_agreement is strong but sparse (w3=0.25)
    - ar_deviation is a weak but safe signal (w4=0.15)

    All components are in [0, 1].  Score is in [0, 1].
    """
    squareness = max(0.0, 1.0 - feats["ar_deviation"])  # clamp at 0
    return (0.40 * min(1.0, feats["color_coverage"] * 5.0)  # scale: 0.20 coverage  1.0
            + 0.20 * feats["edge_symmetry"]
            + 0.25 * min(1.0, feats["cross_branch_iou"] * 2.0)  # IoU 0.50  1.0
            + 0.15 * squareness)


#  Pipeline runner

def process_all_images(
    pipeline_cfg: PipelineConfig,
    sample_images: List[str],
    all_gt: Dict[str, List[Dict]],
    gtsdb_dir: Path,
) -> List[Dict]:
    """Produce per-candidate feature records for all 20 images."""
    all_records = []

    for img_name in sample_images:
        img_bgr = cv2.imread(str(gtsdb_dir / img_name))
        if img_bgr is None:
            continue
        gt_boxes = all_gt.get(img_name, [])
        s1   = run_stage1(img_bgr, pipeline_cfg.stage1)
        enh  = s1["enhanced_image"]
        img_h, img_w = enh.shape[:2]
        gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)

        # Compute raw edges (shared for feature computation)
        blurred  = cv2.GaussianBlur(gray, (5, 5), 1.0)
        edges_raw = cv2.Canny(blurred, 30, 100)

        # Colour masks (for feature computation, not for candidate generation changes)
        color_masks = segment_traffic_sign_colors(enh, pipeline_cfg.stage2)
        red_mask    = color_masks.get("mask_red", np.zeros((img_h, img_w), np.uint8))
        blue_mask   = color_masks.get("mask_blue", np.zeros((img_h, img_w), np.uint8))
        yellow_mask = color_masks.get("mask_yellow", np.zeros((img_h, img_w), np.uint8))

        # Extract candidates from each branch
        color_cands = extract_baseline_color_candidates(enh, pipeline_cfg)
        edge_cands  = extract_edge_candidates_b2(enh, gray, img_h, img_w)

        # Tag source before merge
        tagged = [(b, "color") for b in color_cands] + [(b, "edge") for b in edge_cands]

        # NMS on merged boxes (same as reference pipeline)
        all_boxes = [b for b, _ in tagged]
        post_nms  = non_max_suppression_boxes(all_boxes, pipeline_cfg.stage2.nms_iou_threshold)

        # For each post-NMS box, determine source by finding the pre-NMS entry
        # that matches exactly (NMS returns original box objects)
        color_set = set(color_cands)
        edge_set  = set(edge_cands)

        for box in post_nms:
            if box in color_set:
                src = "color"
                other_branch = edge_cands
            elif box in edge_set:
                src = "edge"
                other_branch = color_cands
            else:
                src = "unknown"
                other_branch = []

            feats = compute_candidate_features(
                box[0], box[1], box[2], box[3],
                gray, edges_raw, red_mask, blue_mask, yellow_mask,
                src, img_h, img_w, other_branch,
            )

            # Label: does this box match any GT at IoU >= 0.50?
            best_iou = 0.0
            best_gt  = -1
            best_cls = -1
            for gi, gt in enumerate(gt_boxes):
                gt_box = (gt["x"], gt["y"], gt["w"], gt["h"])
                iou = compute_iou_xywh(gt_box, box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt  = gi
                    best_cls = gt["class_id"]

            feats["image"]      = img_name
            feats["gt_match"]   = best_iou >= 0.50
            feats["best_gt_iou"] = best_iou
            feats["best_gt_idx"] = best_gt
            feats["best_gt_cls"] = best_cls
            feats["score_c1"]   = score_c1_color_presence(feats)
            feats["score_c2"]   = score_c2_weighted(feats)

            all_records.append(feats)

    return all_records


#  Distribution analysis

def analyse_distributions(records: List[Dict]):
    """Print feature distributions for GT-matching vs non-matching candidates."""
    tp_recs = [r for r in records if r["gt_match"]]
    fp_recs = [r for r in records if not r["gt_match"]]

    print(f"\n{'='*90}")
    print(f"FEATURE DISTRIBUTION ANALYSIS  (TP-matching={len(tp_recs)}, FP={len(fp_recs)})")
    print(f"{'='*90}")

    num_features = [
        "box_area", "aspect_ratio", "ar_deviation", "gray_variance",
        "edge_density", "edge_symmetry", "color_coverage",
        "dominant_color_frac", "center_y_ratio", "cross_branch_iou",
        "score_c1", "score_c2",
    ]

    def stats(vals):
        if not vals:
            return "N/A"
        arr = np.array(vals)
        return (f"mean={np.mean(arr):.4f}  med={np.median(arr):.4f}  "
                f"min={np.min(arr):.4f}  max={np.max(arr):.4f}  "
                f"std={np.std(arr):.4f}")

    for feat in num_features:
        tp_vals = [r[feat] for r in tp_recs]
        fp_vals = [r[feat] for r in fp_recs]
        print(f"\n  {feat}:")
        print(f"    TP: {stats(tp_vals)}")
        print(f"    FP: {stats(fp_vals)}")

    # Source breakdown
    tp_color = sum(1 for r in tp_recs if r["source"] == "color")
    tp_edge  = sum(1 for r in tp_recs if r["source"] == "edge")
    fp_color = sum(1 for r in fp_recs if r["source"] == "color")
    fp_edge  = sum(1 for r in fp_recs if r["source"] == "edge")
    print(f"\n  Source breakdown:")
    print(f"    TP: color={tp_color}, edge={tp_edge}")
    print(f"    FP: color={fp_color}, edge={fp_edge}")

    # Cross-branch agreement
    tp_cross = sum(1 for r in tp_recs if r["cross_branch_iou"] > 0.0)
    fp_cross = sum(1 for r in fp_recs if r["cross_branch_iou"] > 0.0)
    print(f"\n  Cross-branch overlap (IoU > 0):")
    print(f"    TP: {tp_cross}/{len(tp_recs)} ({100*tp_cross/max(1,len(tp_recs)):.1f}%)")
    print(f"    FP: {fp_cross}/{len(fp_recs)} ({100*fp_cross/max(1,len(fp_recs)):.1f}%)")


#  Threshold sweep

def sweep_threshold(
    records: List[Dict],
    score_key: str,
    thresholds: List[float],
    all_gt: Dict[str, List[Dict]],
    label: str,
) -> List[Dict]:
    """Evaluate each score threshold."""
    results = []

    for thresh in thresholds:
        # Group records by image, filter by threshold, evaluate
        by_image: Dict[str, List[Dict]] = {}
        for r in records:
            by_image.setdefault(r["image"], []).append(r)

        total_tp = total_fp = total_fn = 0
        total_gt = 0
        total_post = 0
        per_gt_status = []

        for img_name, img_recs in sorted(by_image.items()):
            surviving = [r for r in img_recs if r[score_key] >= thresh]
            det_boxes = [(r["x"], r["y"], r["w"], r["h"]) for r in surviving]
            gt_boxes  = all_gt.get(img_name, [])
            total_post += len(det_boxes)
            total_gt   += len(gt_boxes)

            ev = evaluate_one_to_one(det_boxes, gt_boxes, 0.50)
            total_tp += ev["tp"]
            total_fp += ev["fp"]
            total_fn += ev["fn"]

            matched_gt = {m["gt_idx"] for m in ev["matches"]}
            for gi, gt in enumerate(gt_boxes):
                per_gt_status.append({
                    "image": img_name, "gt_idx": gi,
                    "class_id": gt["class_id"],
                    "status": "TP" if gi in matched_gt else "FN",
                })

        prec = total_tp / max(1, total_tp + total_fp)
        rec  = total_tp / max(1, total_tp + total_fn)
        f1   = 2 * prec * rec / max(1e-8, prec + rec)

        # Check invariants
        assert total_tp + total_fn == total_gt, f"TP+FN != GT at thresh={thresh}"
        assert total_tp + total_fp == total_post, f"TP+FP != Post at thresh={thresh}"

        # Class 12 status
        c12_signs = [("00008.ppm", 1), ("00010.ppm", 0), ("00016.ppm", 0), ("00019.ppm", 0)]
        c12_ok = sum(1 for s in per_gt_status
                     if (s["image"], s["gt_idx"]) in c12_signs and s["status"] == "TP")
        cls21_ok = any(s["image"] == "00014.ppm" and s["gt_idx"] == 0
                       and s["status"] == "TP" for s in per_gt_status)

        results.append({
            "score_method": label, "threshold": thresh,
            "post_nms": total_post,
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": prec, "recall": rec, "f1": f1,
            "class12_retained": f"{c12_ok}/4",
            "cls21_00014_ok": cls21_ok,
            "per_gt": per_gt_status,
        })

    return results


#  Main

def main():
    root      = Path(__file__).resolve().parent
    ds_cfg    = DatasetConfig()
    gtsdb_dir = root / ds_cfg.gtsdb_dir
    gt_file   = root / ds_cfg.gtsdb_gt_file
    out_dir   = root / "experiments_phase_l"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_images = [f"{i:05d}.ppm" for i in range(20)]
    all_gt, _     = load_gtsdb_annotations(gt_file, gtsdb_dir)

    # K2 + A3 + B2 configuration
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

    #  Step 1: Collect features
    print("Phase L Experiment C  Collecting candidate features")
    records = process_all_images(cfg, sample_images, all_gt, gtsdb_dir)

    # Reproduce reference count
    total_post = len(records)
    tp_count   = sum(1 for r in records if r["gt_match"])
    print(f"\nReference reproduction: post-NMS={total_post}")
    print(f"  Candidates matching any GT at IoU>=0.50: {tp_count}")
    # Note: tp_count can exceed the one-to-one TP count because multiple
    # candidates can overlap the same GT sign.  The one-to-one evaluation
    # resolves this during matching.  Let's run the proper evaluation:
    by_image: Dict[str, List] = {}
    for r in records:
        by_image.setdefault(r["image"], []).append(r)

    ref_tp = ref_fp = ref_fn = 0
    for img_name, recs in sorted(by_image.items()):
        det = [(r["x"], r["y"], r["w"], r["h"]) for r in recs]
        gt  = all_gt.get(img_name, [])
        ev  = evaluate_one_to_one(det, gt, 0.50)
        ref_tp += ev["tp"]; ref_fp += ev["fp"]; ref_fn += ev["fn"]

    print(f"  One-to-one eval: TP={ref_tp} FP={ref_fp} FN={ref_fn} Post={ref_tp+ref_fp}")
    if ref_tp != 17 or ref_fp != 2820:
        print(f"  WARNING: MISMATCH vs expected TP=17 FP=2820. Investigating...")
    else:
        print(f"  OK: Reference reproduced exactly.")

    #  Step 2: Feature analysis
    analyse_distributions(records)

    # Save feature records CSV
    feat_csv = out_dir / "phase_l_expc_features.csv"
    feat_keys = [k for k in records[0].keys() if k not in ("gt_match",)]
    with open(feat_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"\nFeature CSV saved: {feat_csv}")

    #  Step 3: Threshold sweep
    # C1: color coverage thresholds
    c1_thresholds = [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
    c1_results = sweep_threshold(records, "score_c1", c1_thresholds, all_gt, "C1_color_coverage")

    # C2: weighted score thresholds
    c2_thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    c2_results = sweep_threshold(records, "score_c2", c2_thresholds, all_gt, "C2_weighted")

    #  Step 4: Print and save results
    all_sweep = c1_results + c2_results

    c0_tp, c0_fp = ref_tp, ref_fp
    print(f"\n{'='*120}")
    print("THRESHOLD SWEEP  C1 (color coverage) and C2 (weighted score)")
    print(f"{'='*120}")
    hdr = (f"{'Method':<22} | {'Thresh':>6} | {'Post':>5} | {'TP':>2} | "
           f"{'FP':>5} | {'FN':>2} | {'Prec':>7} | {'Rec':>6} | {'F1':>6} | "
           f"{'dTP':>4} | {'dFP':>6} | {'C12':>4}")
    print(hdr)
    print("-" * 120)
    for r in all_sweep:
        dtp = r["tp"] - c0_tp
        dfp = r["fp"] - c0_fp
        print(f"{r['score_method']:<22} | {r['threshold']:>6.3f} | {r['post_nms']:>5} | "
              f"{r['tp']:>2} | {r['fp']:>5} | {r['fn']:>2} | "
              f"{r['precision']:>7.4f} | {r['recall']:>6.4f} | {r['f1']:>6.4f} | "
              f"{dtp:>+4d} | {dfp:>+6d} | {r['class12_retained']}")
    print(f"{'='*120}")

    # Save sweep CSV
    sweep_csv = out_dir / "phase_l_expc_sweep_summary.csv"
    sweep_rows = []
    for r in all_sweep:
        # Identify lost/gained GT signs vs C0
        c0_det = set()
        cur_det = set()
        for s in all_sweep[0]["per_gt"]:
            if s["status"] == "TP":
                c0_det.add(f"{s['image']}:GT#{s['gt_idx']}(cls{s['class_id']})")
        for s in r["per_gt"]:
            if s["status"] == "TP":
                cur_det.add(f"{s['image']}:GT#{s['gt_idx']}(cls{s['class_id']})")
        lost = c0_det - cur_det
        gained = cur_det - c0_det

        sweep_rows.append({
            "score_method": r["score_method"],
            "threshold": r["threshold"],
            "post_nms": r["post_nms"],
            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
            "precision": f"{r['precision']:.4f}",
            "recall": f"{r['recall']:.4f}",
            "f1": f"{r['f1']:.4f}",
            "delta_tp": r["tp"] - c0_tp,
            "delta_fp": r["fp"] - c0_fp,
            "class12_retained": r["class12_retained"],
            "cls21_00014": r["cls21_00014_ok"],
            "lost_vs_c0": "; ".join(sorted(lost)) if lost else "None",
            "gained_vs_c0": "; ".join(sorted(gained)) if gained else "None",
        })

    with open(sweep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader(); w.writerows(sweep_rows)
    print(f"\nSweep CSV: {sweep_csv}")

    # Print which signs are lost at each interesting threshold
    print("\n--- Per-GT changes at key thresholds ---")
    for r in all_sweep:
        if r["tp"] < c0_tp:
            c0_det = set()
            cur_det = set()
            for s in all_sweep[0]["per_gt"]:
                if s["status"] == "TP":
                    c0_det.add(f"{s['image']}:GT#{s['gt_idx']}(cls{s['class_id']})")
            for s in r["per_gt"]:
                if s["status"] == "TP":
                    cur_det.add(f"{s['image']}:GT#{s['gt_idx']}(cls{s['class_id']})")
            lost = c0_det - cur_det
            print(f"  {r['score_method']} thresh={r['threshold']:.3f}: "
                  f"TP={r['tp']} (lost: {'; '.join(sorted(lost))})")


if __name__ == "__main__":
    main()
