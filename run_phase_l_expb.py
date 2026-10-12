"""Phase L — Experiment B: Edge Minimum-Area Ablation.

Tests the effect of raising the edge branch's minimum contour area threshold
on false-positive count and true-positive retention.

Baseline: K2 + A3 traffic-light filter (TP=17, FP=5,283).
Conditions: B0 (min_area=80, reference), B1 (200), B2 (300), B3 (500), B4 (700), B5 (800).

Only the edge min_area parameter is changed. All other settings are identical.

Outputs: experiments_phase_l/phase_l_expb_*.csv
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from src.config import PipelineConfig, DatasetConfig
from src.pipeline import run_stage1
from src.stage2_segmentation.morphology import (
    _is_traffic_light, non_max_suppression_boxes
)
from evaluate_and_compare import load_gtsdb_annotations, compute_iou_xywh, draw_overlay

# Reuse color candidate extraction from Experiment A script
from run_phase_l_expa import extract_baseline_color_candidates, evaluate_one_to_one


# ─── Edge candidates with configurable min_area + A3 traffic-light filter ──────

def extract_edge_candidates_expb(
    enhanced_bgr: np.ndarray,
    image_gray: np.ndarray,
    img_h: int,
    img_w: int,
    min_area: float = 80.0,
    gt_boxes: List[Dict] = None,
) -> Tuple[List[Tuple[int, int, int, int]], Dict[str, int], List[Dict]]:
    """
    Generate edge candidates with configurable min_area + A3 traffic-light filter.

    Returns:
        candidates: accepted (x, y, w, h) tuples.
        rejection_log: counts per rejection reason.
        area_rejected_gt_hits: list of dicts for area-rejected candidates that
            would have matched a GT sign at IoU >= 0.50.
    """
    blurred  = cv2.GaussianBlur(image_gray, (5, 5), 1.0)
    edges    = cv2.Canny(blurred, 30, 100)
    k        = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_cl = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(edges_cl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rej = {
        "area_below_min": 0, "area_above_max": 0,
        "box_dim": 0, "spatial": 0, "aspect": 0, "symmetry": 0,
        "traffic_light": 0, "accepted": 0,
    }

    candidates = []
    area_rejected_gt_hits = []  # track lost GT matches from area filter

    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)

        # Area gate — split into below-min and above-max for diagnostics
        if cnt_area < min_area:
            # Check if this would have matched any GT at IoU >= 0.50
            if gt_boxes:
                x, y, w, h = cv2.boundingRect(cnt)
                for gi, gt in enumerate(gt_boxes):
                    gt_box = (gt["x"], gt["y"], gt["w"], gt["h"])
                    iou = compute_iou_xywh(gt_box, (x, y, w, h))
                    if iou >= 0.50:
                        area_rejected_gt_hits.append({
                            "gt_idx": gi, "class_id": gt["class_id"],
                            "contour_area": cnt_area, "box": (x, y, w, h),
                            "iou": iou, "min_area_threshold": min_area,
                        })
            rej["area_below_min"] += 1
            continue
        if cnt_area > 10000.0:
            rej["area_above_max"] += 1
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 15 or h < 15 or w > 200 or h > 200:
            rej["box_dim"] += 1
            continue
        if y > int(0.95 * img_h) or (y + h) < int(0.04 * img_h):
            rej["spatial"] += 1
            continue
        ar = float(w) / max(1, h)
        if not (0.45 <= ar <= 1.60):
            rej["aspect"] += 1
            continue

        # K2 symmetry gate
        crop_e    = edges[y:y+h, x:x+w]
        crop_flip = cv2.flip(crop_e, 1)
        sym = np.sum(crop_e & crop_flip) / max(1.0, np.sum(crop_e))
        if sym < 0.15:
            rej["symmetry"] += 1
            continue

        # A3 traffic-light filter (always on, zero TP cost from Exp A)
        if _is_traffic_light(image_gray, x, y, w, h):
            rej["traffic_light"] += 1
            continue

        rej["accepted"] += 1
        candidates.append((x, y, w, h))

    return candidates, rej, area_rejected_gt_hits


# ─── Per-condition runner ──────────────────────────────────────────────────────

def run_expb_condition(
    cond_name: str,
    edge_min_area: float,
    pipeline_cfg: PipelineConfig,
    sample_images: List[str],
    all_gt: Dict[str, List[Dict]],
    gtsdb_dir: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """Execute one Experiment B condition."""
    cond_dir = out_dir / cond_name
    cond_dir.mkdir(parents=True, exist_ok=True)
    (cond_dir / "overlays").mkdir(exist_ok=True)

    per_image_rows = []
    per_gt_rows    = []
    totals = dict(gt=0, pre_nms=0, post_nms=0, tp=0, fp=0, fn=0,
                  rej_area_below=0, rej_tl=0)
    all_area_gt_hits = []

    print("=" * 90)
    print(f"CONDITION: {cond_name}  [edge_min_area={edge_min_area}]")
    print("=" * 90)

    for idx, img_name in enumerate(sample_images, 1):
        img_bgr = cv2.imread(str(gtsdb_dir / img_name))
        if img_bgr is None:
            continue
        gt_boxes = all_gt.get(img_name, [])
        s1   = run_stage1(img_bgr, pipeline_cfg.stage1)
        enh  = s1["enhanced_image"]
        img_h, img_w = enh.shape[:2]
        gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)

        color_cands = extract_baseline_color_candidates(enh, pipeline_cfg)
        edge_cands, rej, gt_hits = extract_edge_candidates_expb(
            enh, gray, img_h, img_w,
            min_area=edge_min_area, gt_boxes=gt_boxes,
        )

        for hit in gt_hits:
            hit["image"] = img_name
            all_area_gt_hits.append(hit)

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
        totals["rej_area_below"] += rej["area_below_min"]
        totals["rej_tl"]   += rej["traffic_light"]

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
            "image": img_name, "gt_signs": len(gt_boxes),
            "color_cands": len(color_cands), "edge_cands": len(edge_cands),
            "pre_nms": n_pre_nms, "post_nms": n_post_nms,
            "tp": eval_res["tp"], "fp": eval_res["fp"], "fn": eval_res["fn"],
            "precision": f"{eval_res['precision']:.4f}",
            "recall":    f"{eval_res['recall']:.4f}",
            "rej_area_below": rej["area_below_min"],
            "rej_traffic_light": rej["traffic_light"],
        })

        tp_info = ""
        if eval_res["tp"] > 0:
            tp_info = " | " + ", ".join(
                [f"GT#{m['gt_idx']} cls={m['class_id']} (IoU={m['iou']:.2f})"
                 for m in eval_res["matches"]])
        print(f"  [{idx:2d}/20] {img_name} Col={len(color_cands):3d} Edge={len(edge_cands):3d} "
              f"Post={n_post_nms:4d} TP={eval_res['tp']} FP={eval_res['fp']}{tp_info}")

    # Write per-image and GT status CSVs
    with open(cond_dir / f"{cond_name}_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        w.writeheader(); w.writerows(per_image_rows)

    with open(cond_dir / f"{cond_name}_gt_status.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_gt_rows[0].keys()))
        w.writeheader(); w.writerows(per_gt_rows)

    prec = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    rec  = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    f1   = 2 * prec * rec / max(1e-8, prec + rec)

    assert totals["tp"] + totals["fn"] == totals["gt"], \
        f"TP+FN={totals['tp']+totals['fn']} != GT={totals['gt']}"
    assert totals["tp"] + totals["fp"] == totals["post_nms"], \
        f"TP+FP={totals['tp']+totals['fp']} != PostNMS={totals['post_nms']}"

    print(f"\n  [{cond_name}] GT={totals['gt']} Pre={totals['pre_nms']} Post={totals['post_nms']} "
          f"TP={totals['tp']} FP={totals['fp']} FN={totals['fn']} "
          f"P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    print(f"  Edge area rejections: {totals['rej_area_below']}  "
          f"Traffic-light rejections: {totals['rej_tl']}\n")

    return {
        "condition": cond_name, "edge_min_area": edge_min_area,
        "pre_nms": totals["pre_nms"], "post_nms": totals["post_nms"],
        "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"],
        "precision": prec, "recall": rec, "f1": f1,
        "rej_area_below": totals["rej_area_below"],
        "rej_traffic_light": totals["rej_tl"],
        "matches": [r for r in per_gt_rows if r["status"] == "DETECTED_TP"],
        "area_rejected_gt_hits": all_area_gt_hits,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    root      = Path(__file__).resolve().parent
    ds_cfg    = DatasetConfig()
    gtsdb_dir = root / ds_cfg.gtsdb_dir
    gt_file   = root / ds_cfg.gtsdb_gt_file
    out_dir   = root / "experiments_phase_l"
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
        ("B0_MinArea_80",  80.0),
        ("B1_MinArea_200", 200.0),
        ("B2_MinArea_300", 300.0),
        ("B3_MinArea_500", 500.0),
        ("B4_MinArea_700", 700.0),
        ("B5_MinArea_800", 800.0),
    ]

    results = []
    for cond_name, min_area in conditions:
        res = run_expb_condition(
            cond_name, min_area, cfg,
            sample_images, all_gt, gtsdb_dir, out_dir,
        )
        results.append(res)

    # ─── Summary Table ─────────────────────────────────────────────────────────
    b0 = results[0]
    b0_tp = b0["tp"]
    b0_fp = b0["fp"]
    b0_det = {f"{m['image']}:GT#{m['gt_idx']}(cls{m['class_id']})" for m in b0["matches"]}
    CLASS12 = {"00008.ppm:GT#1(cls12)", "00010.ppm:GT#0(cls12)",
               "00016.ppm:GT#0(cls12)", "00019.ppm:GT#0(cls12)"}

    print("\n" + "=" * 130)
    print("PHASE L EXPERIMENT B — SUMMARY  (IoU >= 0.50, one-to-one matching)")
    print("=" * 130)
    hdr = (f"{'Condition':<20} | {'MinA':>4} | {'PreNMS':>6} | {'Post':>5} | "
           f"{'TP':>2} | {'FP':>5} | {'FN':>2} | {'Prec':>7} | {'Rec':>6} | {'F1':>6} | "
           f"{'dTP':>4} | {'dFP':>6} | {'C12':>4} | {'AreaRej':>7} | {'GT-hits-lost':>12}")
    print(hdr)
    print("-" * 130)

    summary_rows = []
    for r in results:
        cur_det  = {f"{m['image']}:GT#{m['gt_idx']}(cls{m['class_id']})" for m in r["matches"]}
        gained   = cur_det - b0_det
        lost     = b0_det - cur_det
        c12_kept = len(CLASS12 & cur_det)
        n_gt_hits = len(r["area_rejected_gt_hits"])

        row = {
            "condition": r["condition"], "edge_min_area": int(r["edge_min_area"]),
            "pre_nms": r["pre_nms"], "post_nms": r["post_nms"],
            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
            "precision": f"{r['precision']:.4f}",
            "recall":    f"{r['recall']:.4f}",
            "f1":        f"{r['f1']:.4f}",
            "delta_tp_vs_b0": f"{r['tp'] - b0_tp:+d}",
            "delta_fp_vs_b0": f"{r['fp'] - b0_fp:+d}",
            "class12_retained": f"{c12_kept}/4",
            "edge_area_rejections": r["rej_area_below"],
            "area_rejected_gt_hits": n_gt_hits,
            "gained_vs_b0": "; ".join(sorted(gained)) if gained else "None",
            "lost_vs_b0":   "; ".join(sorted(lost))   if lost   else "None",
            "detected_signs": "; ".join(sorted(cur_det)),
        }
        summary_rows.append(row)

        dtp = f"{r['tp'] - b0_tp:+d}"
        dfp = f"{r['fp'] - b0_fp:+d}"
        print(f"{r['condition']:<20} | {int(r['edge_min_area']):>4} | {r['pre_nms']:>6} | "
              f"{r['post_nms']:>5} | {r['tp']:>2} | {r['fp']:>5} | {r['fn']:>2} | "
              f"{r['precision']:>7.4f} | {r['recall']:>6.4f} | {r['f1']:>6.4f} | "
              f"{dtp:>4} | {dfp:>6} | {c12_kept:>2}/4 | "
              f"{r['rej_area_below']:>7} | {n_gt_hits:>12}")
        if lost:
            print(f"  ** LOST: {'; '.join(sorted(lost))}")
        if gained:
            print(f"  ++ GAINED: {'; '.join(sorted(gained))}")

    print("=" * 130)

    # Write summary CSV
    summary_csv = out_dir / "phase_l_expb_comparison_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"\nSummary CSV: {summary_csv}")

    # Write area-rejected GT hits diagnostic
    all_gt_hits = []
    for r in results:
        all_gt_hits.extend(r["area_rejected_gt_hits"])
    if all_gt_hits:
        hits_csv = out_dir / "phase_l_expb_area_rejected_gt_hits.csv"
        with open(hits_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_gt_hits[0].keys()))
            w.writeheader(); w.writerows(all_gt_hits)
        print(f"GT-hits diagnostic CSV: {hits_csv}")

        print("\n--- Area-rejected edge candidates that matched GT (IoU >= 0.50) ---")
        for h in all_gt_hits:
            print(f"  {h['image']} GT#{h['gt_idx']} cls={h['class_id']}: "
                  f"contour_area={h['contour_area']:.0f} IoU={h['iou']:.4f} "
                  f"threshold={h['min_area_threshold']}")
    else:
        print("\nNo area-rejected edge candidates would have matched any GT sign.")


if __name__ == "__main__":
    main()
