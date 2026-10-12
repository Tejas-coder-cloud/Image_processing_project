"""Investigation and evaluation runner for Stage 1-3 traffic sign detection.

Supports:
- Evaluation on GTSDB dataset using ground-truth gt.txt format (filename;left;top;right;bottom;class_id)
- One-to-one IoU matching at IoU threshold = 0.5 (or configurable)
- Baseline run vs Experiment 1 (vegetation-mask subtraction disabled)
- Generation of side-by-side comparison overlays and comprehensive CSV summaries
- Clean preservation of previous results without overwriting original data
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.config import PipelineConfig, DatasetConfig
from src.pipeline import run_full_pipeline


def load_gtsdb_annotations(
    gt_file: Path,
    img_dir: Path,
    expected_width: int = 1360,
    expected_height: int = 800,
) -> Tuple[Dict[str, List[Dict]], Dict[str, Any]]:
    """Load GTSDB ground truth annotations from gt.txt and convert boxes to common x1, y1, x2, y2 format.

    Format: filename;left;top;right;bottom;class_id
    Coordinates:
        x1 = left
        y1 = top
        x2 = right
        y2 = bottom
        w = x2 - x1
        h = y2 - y1

    Validates boxes against image boundaries and dimensions.
    Returns:
        (annotations_by_image, validation_summary)
    """
    annotations_by_image: Dict[str, List[Dict]] = {}
    validation_summary = {
        "total_lines": 0,
        "valid_boxes": 0,
        "invalid_boxes": [],
        "out_of_bounds_boxes": [],
        "missing_images": [],
        "images_with_annotations": set(),
        "empty_label_images": [],
    }

    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

    with open(gt_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    validation_summary["total_lines"] = len(lines)

    dims_cache: Dict[str, Tuple[int, int]] = {}

    for line_idx, line in enumerate(lines, 1):
        parts = line.split(";")
        if len(parts) < 6:
            validation_summary["invalid_boxes"].append((line_idx, line, "Fewer than 6 fields"))
            continue

        img_name, left_s, top_s, right_s, bottom_s, cls_s = parts[:6]
        img_path = img_dir / img_name

        if img_name not in dims_cache:
            if img_path.exists():
                im = cv2.imread(str(img_path))
                if im is not None:
                    dims_cache[img_name] = (im.shape[1], im.shape[0])  # (width, height)
                else:
                    dims_cache[img_name] = (expected_width, expected_height)
            else:
                validation_summary["missing_images"].append(img_name)
                dims_cache[img_name] = (expected_width, expected_height)

        img_w, img_h = dims_cache[img_name]

        try:
            x1 = int(left_s)
            y1 = int(top_s)
            x2 = int(right_s)
            y2 = int(bottom_s)
            class_id = int(cls_s)
        except ValueError as e:
            validation_summary["invalid_boxes"].append((line_idx, line, f"Parse error: {e}"))
            continue

        # Common representation: x1, y1, x2, y2, w, h
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            validation_summary["invalid_boxes"].append(
                (line_idx, line, f"Degenerate box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            )
            continue

        if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
            validation_summary["out_of_bounds_boxes"].append(
                (line_idx, line, f"Out of bounds: box=({x1},{y1},{x2},{y2}), img=({img_w},{img_h})")
            )
            # Clip bounds safely to image boundaries
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            w = x2 - x1
            h = y2 - y1

        entry = {
            "image_name": img_name,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "x": x1,
            "y": y1,
            "w": w,
            "h": h,
            "class_id": class_id,
        }

        annotations_by_image.setdefault(img_name, []).append(entry)
        validation_summary["images_with_annotations"].add(img_name)
        validation_summary["valid_boxes"] += 1

    # Check images without annotations (empty labels / negative examples)
    all_images = [f.name for f in img_dir.glob("*.ppm")]
    for img_name in all_images:
        if img_name not in validation_summary["images_with_annotations"]:
            validation_summary["empty_label_images"].append(img_name)

    return annotations_by_image, validation_summary


def compute_iou_xywh(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    union_area = w1 * h1 + w2 * h2 - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area) / union_area


def evaluate_one_to_one(
    detected_bboxes: List[Tuple[int, int, int, int]],
    gt_bboxes: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Perform strict one-to-one matching between detected boxes and ground-truth boxes.

    Greedy assignment by highest IoU above iou_threshold.
    Returns:
        Dict containing TP, FP, FN, precision, recall, f1, matches, unmatched indices.
    """
    n_gt = len(gt_bboxes)
    n_det = len(detected_bboxes)

    # Compute pairwise IoU matrix
    iou_pairs = []
    for gi, gt in enumerate(gt_bboxes):
        gt_box = (gt["x"], gt["y"], gt["w"], gt["h"])
        for di, det_box in enumerate(detected_bboxes):
            iou = compute_iou_xywh(gt_box, det_box)
            if iou >= iou_threshold:
                iou_pairs.append((iou, gi, di))

    # Sort descending by IoU for optimal greedy matching
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


def draw_overlay(
    image_bgr: np.ndarray,
    detected_bboxes: List[Tuple[int, int, int, int]],
    gt_bboxes: List[Dict],
    eval_result: Dict[str, Any],
    title: str = "",
) -> np.ndarray:
    """Render annotated visualization showing TP (green), FP (red), and FN (cyan)."""
    overlay = image_bgr.copy()
    h, w = overlay.shape[:2]

    # TP in bright green
    for match in eval_result["matches"]:
        di = match["det_idx"]
        x, y, bw, bh = detected_bboxes[di]
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 220, 0), 2)
        label = f"TP ({match['iou']:.2f})"
        cv2.putText(overlay, label, (x, max(14, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 0), 1, cv2.LINE_AA)

    # FP in red
    for di in eval_result["unmatched_det_indices"]:
        x, y, bw, bh = detected_bboxes[di]
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(overlay, "FP", (x, max(14, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1, cv2.LINE_AA)

    # FN (missed ground truth) in dashed cyan
    for gi in eval_result["unmatched_gt_indices"]:
        gt = gt_bboxes[gi]
        x, y, bw, bh = gt["x"], gt["y"], gt["w"], gt["h"]
        for i in range(0, bw, 8):
            x1 = x + i
            x2 = min(x + i + 4, x + bw)
            cv2.line(overlay, (x1, y), (x2, y), (255, 255, 0), 2)
            cv2.line(overlay, (x1, y + bh), (x2, y + bh), (255, 255, 0), 2)
        for i in range(0, bh, 8):
            y1 = y + i
            y2 = min(y + i + 4, y + bh)
            cv2.line(overlay, (x, y1), (x, y2), (255, 255, 0), 2)
            cv2.line(overlay, (x + bw, y1), (x + bw, y2), (255, 255, 0), 2)
        label = f"MISS:cls{gt['class_id']}"
        cv2.putText(overlay, label, (x, max(14, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)

    # Header / stats banner
    banner_text = (
        f"{title} | TP={eval_result['tp']} FP={eval_result['fp']} FN={eval_result['fn']} "
        f"P={eval_result['precision']:.2f} R={eval_result['recall']:.2f}"
    )
    cv2.rectangle(overlay, (0, h - 28), (w, h), (15, 23, 42), -1)
    cv2.putText(overlay, banner_text, (10, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (241, 245, 249), 1, cv2.LINE_AA)

    return overlay


def create_side_by_side(
    img_baseline: np.ndarray,
    img_exp1: np.ndarray,
    img_name: str,
) -> np.ndarray:
    """Concatenate baseline and Exp1 overlays side-by-side with top labeling."""
    h, w = img_baseline.shape[:2]
    combined = np.hstack([img_baseline, img_exp1])

    # Add header banner
    banner = np.zeros((36, combined.shape[1], 3), dtype=np.uint8)
    banner[:] = (30, 41, 59)
    cv2.putText(banner, f"Image: {img_name} | Left: BASELINE (with veg subtraction) | Right: EXP 1 (without veg subtraction)",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([banner, combined])


def run_experiment_suite(
    sample_images: List[str],
    gtsdb_dir: Path,
    gt_file: Path,
    results_dir: Path,
    iou_thresh: float = 0.5,
):
    """Run baseline and experiment 1, produce side-by-side images and summary CSVs."""
    results_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = results_dir / "baseline"
    exp1_dir = results_dir / "experiment_1"
    vis_dir = results_dir / "side_by_side"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    exp1_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 1: VALIDATING GTSDB ANNOTATIONS")
    print("=" * 80)
    all_gt, val_stats = load_gtsdb_annotations(gt_file, gtsdb_dir)
    print(f"Total annotations in gt.txt: {val_stats['total_lines']}")
    print(f"Valid sign bounding boxes: {val_stats['valid_boxes']}")
    print(f"Images with annotations: {len(val_stats['images_with_annotations'])}")
    print(f"Images with empty labels: {len(val_stats['empty_label_images'])}")
    print(f"Invalid / degenerate boxes: {len(val_stats['invalid_boxes'])}")
    print(f"Out of bounds boxes: {len(val_stats['out_of_bounds_boxes'])}")
    print(f"Missing image references: {len(val_stats['missing_images'])}\n")

    cfg_baseline = PipelineConfig()
    cfg_baseline.stage2.subtract_vegetation = True

    cfg_exp1 = PipelineConfig()
    cfg_exp1.stage2.subtract_vegetation = False

    rows_baseline = []
    rows_exp1 = []
    rows_comparison = []

    print("=" * 80)
    print(f"STEP 2: RUNNING BASELINE & EXPERIMENT 1 ON {len(sample_images)} REPRESENTATIVE IMAGES")
    print(f"Evaluation IoU threshold: {iou_thresh}")
    print("=" * 80)

    for idx, img_name in enumerate(sample_images, 1):
        img_path = gtsdb_dir / img_name
        if not img_path.exists():
            print(f"[{idx}/{len(sample_images)}] Error: {img_name} not found")
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[{idx}/{len(sample_images)}] Error: could not load {img_name}")
            continue

        gt_boxes = all_gt.get(img_name, [])

        # 1. Baseline Run
        res_base = run_full_pipeline(img_bgr, cfg_baseline)
        base_det = res_base["stage2"]["bounding_boxes"]
        base_eval = evaluate_one_to_one(base_det, gt_boxes, iou_threshold=iou_thresh)
        base_overlay = draw_overlay(
            res_base["stage1"]["enhanced_image"], base_det, gt_boxes, base_eval, "BASELINE"
        )

        cv2.imwrite(str(baseline_dir / f"{Path(img_name).stem}_overlay.png"), base_overlay)
        cv2.imwrite(str(baseline_dir / f"{Path(img_name).stem}_mask.png"), res_base["stage2"]["binary_mask_cleaned"])

        rows_baseline.append({
            "image": img_name,
            "gt_signs": len(gt_boxes),
            "detected": len(base_det),
            "tp": base_eval["tp"],
            "fp": base_eval["fp"],
            "fn": base_eval["fn"],
            "precision": f"{base_eval['precision']:.4f}",
            "recall": f"{base_eval['recall']:.4f}",
            "f1": f"{base_eval['f1']:.4f}",
        })

        # 2. Experiment 1 Run (No Vegetation Subtraction)
        res_exp1 = run_full_pipeline(img_bgr, cfg_exp1)
        exp1_det = res_exp1["stage2"]["bounding_boxes"]
        exp1_eval = evaluate_one_to_one(exp1_det, gt_boxes, iou_threshold=iou_thresh)
        exp1_overlay = draw_overlay(
            res_exp1["stage1"]["enhanced_image"], exp1_det, gt_boxes, exp1_eval, "EXP 1 (NO VEG SUB)"
        )

        cv2.imwrite(str(exp1_dir / f"{Path(img_name).stem}_overlay.png"), exp1_overlay)
        cv2.imwrite(str(exp1_dir / f"{Path(img_name).stem}_mask.png"), res_exp1["stage2"]["binary_mask_cleaned"])

        rows_exp1.append({
            "image": img_name,
            "gt_signs": len(gt_boxes),
            "detected": len(exp1_det),
            "tp": exp1_eval["tp"],
            "fp": exp1_eval["fp"],
            "fn": exp1_eval["fn"],
            "precision": f"{exp1_eval['precision']:.4f}",
            "recall": f"{exp1_eval['recall']:.4f}",
            "f1": f"{exp1_eval['f1']:.4f}",
        })

        # Side-by-side visualization
        sbs = create_side_by_side(base_overlay, exp1_overlay, img_name)
        cv2.imwrite(str(vis_dir / f"{Path(img_name).stem}_comparison.png"), sbs)

        # Comparison row
        rows_comparison.append({
            "image": img_name,
            "gt_signs": len(gt_boxes),
            "baseline_det": len(base_det),
            "baseline_tp": base_eval["tp"],
            "baseline_fp": base_eval["fp"],
            "baseline_fn": base_eval["fn"],
            "baseline_prec": f"{base_eval['precision']:.4f}",
            "baseline_rec": f"{base_eval['recall']:.4f}",
            "baseline_f1": f"{base_eval['f1']:.4f}",
            "exp1_det": len(exp1_det),
            "exp1_tp": exp1_eval["tp"],
            "exp1_fp": exp1_eval["fp"],
            "exp1_fn": exp1_eval["fn"],
            "exp1_prec": f"{exp1_eval['precision']:.4f}",
            "exp1_rec": f"{exp1_eval['recall']:.4f}",
            "exp1_f1": f"{exp1_eval['f1']:.4f}",
            "tp_delta": exp1_eval["tp"] - base_eval["tp"],
            "fp_delta": exp1_eval["fp"] - base_eval["fp"],
        })

        print(
            f"[{idx:2d}/20] {img_name:12s} | GT={len(gt_boxes):2d} | "
            f"Base: Det={len(base_det):2d} TP={base_eval['tp']} FP={base_eval['fp']:2d} FN={base_eval['fn']} | "
            f"Exp1: Det={len(exp1_det):2d} TP={exp1_eval['tp']} FP={exp1_eval['fp']:2d} FN={exp1_eval['fn']}"
        )

    # Save CSVs
    def save_csv(path: Path, rows: List[Dict]):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    save_csv(baseline_dir / "baseline_summary.csv", rows_baseline)
    save_csv(exp1_dir / "experiment_1_summary.csv", rows_exp1)
    save_csv(results_dir / "comparison_summary.csv", rows_comparison)

    # Overall totals
    def aggregate(rows: List[Dict], prefix: str = ""):
        total_gt = sum(int(r[f"{prefix}gt_signs" if prefix else "gt_signs"]) for r in rows)
        total_det = sum(int(r[f"{prefix}det" if prefix else "detected"]) for r in rows)
        total_tp = sum(int(r[f"{prefix}tp"]) for r in rows)
        total_fp = sum(int(r[f"{prefix}fp"]) for r in rows)
        total_fn = sum(int(r[f"{prefix}fn"]) for r in rows)
        p = total_tp / max(1, total_tp + total_fp)
        r = total_tp / max(1, total_tp + total_fn)
        f1 = 2 * p * r / max(1e-8, p + r)
        return total_gt, total_det, total_tp, total_fp, total_fn, p, r, f1

    b_gt, b_det, b_tp, b_fp, b_fn, b_p, b_r, b_f1 = aggregate(rows_baseline)
    e_gt, e_det, e_tp, e_fp, e_fn, e_p, e_r, e_f1 = aggregate(rows_exp1)

    print("\n" + "=" * 80)
    print("EXPERIMENT 1 vs BASELINE METRIC SUMMARY (IoU >= 0.5)")
    print("=" * 80)
    print(f"{'Metric':<20} | {'Baseline':<15} | {'Experiment 1':<15} | {'Delta':<15}")
    print("-" * 72)
    print(f"{'Total GT Signs':<20} | {b_gt:<15d} | {e_gt:<15d} | {e_gt - b_gt:<15d}")
    print(f"{'Candidate Detections':<20} | {b_det:<15d} | {e_det:<15d} | {e_det - b_det:<15d}")
    print(f"{'True Positives (TP)':<20} | {b_tp:<15d} | {e_tp:<15d} | {e_tp - b_tp:<15d}")
    print(f"{'False Positives (FP)':<20} | {b_fp:<15d} | {e_fp:<15d} | {e_fp - b_fp:<15d}")
    print(f"{'False Negatives (FN)':<20} | {b_fn:<15d} | {e_fn:<15d} | {e_fn - b_fn:<15d}")
    print(f"{'Precision':<20} | {b_p:<15.4f} | {e_p:<15.4f} | {e_p - b_p:<+15.4f}")
    print(f"{'Recall':<20} | {b_r:<15.4f} | {e_r:<15.4f} | {e_r - b_r:<+15.4f}")
    print(f"{'F1 Score':<20} | {b_f1:<15.4f} | {e_f1:<15.4f} | {e_f1 - b_f1:<+15.4f}")
    print("=" * 80)
    print(f"Results successfully saved to: {results_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Traffic Sign Pipeline: Baseline vs Exp 1")
    parser.add_argument(
        "--gtsdb_dir",
        type=str,
        default=str(DatasetConfig().gtsdb_dir),
        help="Path to GTSDB images directory containing .ppm files",
    )
    parser.add_argument(
        "--gt_file",
        type=str,
        default=str(DatasetConfig().gtsdb_gt_file),
        help="Path to gt.txt annotation file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="new_investigation_results",
        help="Output directory for new baseline and Experiment 1 results",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for one-to-one ground truth matching (default: 0.5)",
    )

    args = parser.parse_args()

    # 20 representative sample images
    sample_images = [f"{i:05d}.ppm" for i in range(20)]

    run_experiment_suite(
        sample_images=sample_images,
        gtsdb_dir=Path(args.gtsdb_dir),
        gt_file=Path(args.gt_file),
        results_dir=Path(args.output_dir),
        iou_thresh=args.iou_threshold,
    )


if __name__ == "__main__":
    main()
