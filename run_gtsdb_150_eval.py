"""Batch evaluation on 150 GTSDB images with ground truth comparison.

Runs the full Stage 1 + Stage 2 pipeline on 150 diverse GTSDB images,
compares detections against ground truth labels, and generates:
1. Per-image annotated overlays (detected bboxes + GT bboxes)
2. Detection statistics (true positives, false positives, missed signs)
3. A master gallery grid
4. A CSV summary report

Dataset label format (YOLO):  class_id  cx_norm  cy_norm  w_norm  h_norm
Classes: 0=danger, 1=mandatory, 2=other, 3=prohibitory
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import PipelineConfig, DatasetConfig
from src.pipeline import run_full_pipeline
from src.visualizer import plot_pipeline_summary, plot_stage1_report, plot_stage2_report


CLASS_NAMES = {0: "danger", 1: "mandatory", 2: "other", 3: "prohibitory"}
CLASS_COLORS = {
    0: (0, 0, 255),    # red (BGR) for danger
    1: (255, 150, 0),   # blue for mandatory
    2: (0, 200, 200),   # yellow for other
    3: (0, 0, 200),     # dark red for prohibitory
}


def load_ground_truth(label_path: Path, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """Load YOLO format ground truth labels and convert to pixel coordinates.

    Returns list of dicts with keys: class_id, class_name, x, y, w, h (pixel coords).
    """
    gt_list = []
    if not label_path.exists():
        return gt_list

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx_norm = float(parts[1])
            cy_norm = float(parts[2])
            bw_norm = float(parts[3])
            bh_norm = float(parts[4])

            # Convert normalized center coords to pixel top-left coords
            bw_px = int(bw_norm * img_w)
            bh_px = int(bh_norm * img_h)
            bx_px = int(cx_norm * img_w - bw_px / 2)
            by_px = int(cy_norm * img_h - bh_px / 2)

            gt_list.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, f"cls_{class_id}"),
                "x": max(0, bx_px),
                "y": max(0, by_px),
                "w": bw_px,
                "h": bh_px,
            })

    return gt_list


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
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


def evaluate_detections(
    detected_bboxes: List[Tuple[int, int, int, int]],
    gt_bboxes: List[Dict[str, Any]],
    iou_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Match detected bboxes to ground truth using IoU and compute TP/FP/FN.

    Uses a greedy matching strategy: for each GT, find best matching detection.
    IoU threshold is set to 0.25 (lenient) since color-based detection may not
    perfectly align with GT boundaries.
    """
    n_gt = len(gt_bboxes)
    n_det = len(detected_bboxes)

    matched_gt = set()
    matched_det = set()
    matches = []

    # Build IoU matrix
    for gi, gt in enumerate(gt_bboxes):
        gt_box = (gt["x"], gt["y"], gt["w"], gt["h"])
        best_iou = 0.0
        best_di = -1
        for di, det_box in enumerate(detected_bboxes):
            if di in matched_det:
                continue
            iou = compute_iou(gt_box, det_box)
            if iou > best_iou:
                best_iou = iou
                best_di = di

        if best_iou >= iou_threshold and best_di >= 0:
            matched_gt.add(gi)
            matched_det.add(best_di)
            matches.append({
                "gt_idx": gi,
                "det_idx": best_di,
                "iou": best_iou,
                "class_name": gt["class_name"],
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


def draw_evaluation_overlay(
    image_bgr: np.ndarray,
    detected_bboxes: List[Tuple[int, int, int, int]],
    gt_bboxes: List[Dict[str, Any]],
    eval_result: Dict[str, Any],
) -> np.ndarray:
    """Draw annotated overlay showing:
    - Green boxes: True Positive detections (matched GT)
    - Red boxes: False Positive detections (no GT match)
    - Cyan dashed boxes: Missed GT signs (False Negatives)
    """
    overlay = image_bgr.copy()
    h, w = overlay.shape[:2]

    # Draw matched detections (True Positives) in GREEN
    for match in eval_result["matches"]:
        di = match["det_idx"]
        x, y, bw, bh = detected_bboxes[di]
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 220, 0), 2)
        label = f"TP ({match['iou']:.2f})"
        cv2.putText(overlay, label, (x, max(12, y - 4)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 0), 1, cv2.LINE_AA)

    # Draw false positive detections in RED
    for di in eval_result["unmatched_det_indices"]:
        x, y, bw, bh = detected_bboxes[di]
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(overlay, "FP", (x, max(12, y - 4)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1, cv2.LINE_AA)

    # Draw missed ground truth signs in CYAN (dashed style via small segments)
    for gi in eval_result["unmatched_gt_indices"]:
        gt = gt_bboxes[gi]
        x, y, bw, bh = gt["x"], gt["y"], gt["w"], gt["h"]
        # Draw dashed rectangle
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
        label = f"MISS:{gt['class_name']}"
        cv2.putText(overlay, label, (x, max(12, y - 4)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

    # Draw stats bar at bottom
    stats_text = f"TP={eval_result['tp']}  FP={eval_result['fp']}  FN={eval_result['fn']}  P={eval_result['precision']:.2f}  R={eval_result['recall']:.2f}"
    bar_y = h - 25
    cv2.rectangle(overlay, (0, bar_y - 5), (w, h), (0, 0, 0), -1)
    cv2.putText(overlay, stats_text, (8, bar_y + 12),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def select_150_images(gtsdb_dir: Path) -> List[Tuple[Path, Path]]:
    """Select 150 diverse GTSDB images, preferring unique scene IDs.

    Returns list of (image_path, label_path) tuples.
    """
    images_dir = gtsdb_dir / "train" / "images"
    labels_dir = gtsdb_dir / "train" / "labels"

    all_images = sorted(images_dir.glob("*.jpg"))
    if not all_images:
        all_images = sorted(images_dir.glob("*.png"))

    # Group by scene ID (e.g., "100_jpg" from "100_jpg.rf.xxx.jpg")
    scene_groups: Dict[str, List[Path]] = {}
    for img_path in all_images:
        scene_id = img_path.name.split(".rf.")[0] if ".rf." in img_path.name else img_path.stem
        scene_groups.setdefault(scene_id, []).append(img_path)

    selected: List[Tuple[Path, Path]] = []

    # First pass: one image per scene to maximize diversity
    scene_ids = sorted(scene_groups.keys())
    for sid in scene_ids:
        if len(selected) >= 150:
            break
        img_path = scene_groups[sid][0]
        label_name = img_path.stem + ".txt"
        label_path = labels_dir / label_name
        selected.append((img_path, label_path))

    # If we have fewer than 150 unique scenes, take more augmentations
    if len(selected) < 150:
        remaining_pool = []
        for sid in scene_ids:
            for img_path in scene_groups[sid][1:]:
                label_name = img_path.stem + ".txt"
                label_path = labels_dir / label_name
                remaining_pool.append((img_path, label_path))
        random.seed(42)
        random.shuffle(remaining_pool)
        for pair in remaining_pool:
            if len(selected) >= 150:
                break
            selected.append(pair)

    return selected[:150]


def generate_summary_gallery(
    results: List[Dict[str, Any]],
    save_path: Path,
    max_rows: int = 20,
):
    """Generate a master comparison gallery showing detection results vs ground truth."""
    n_show = min(len(results), max_rows)
    fig, axes = plt.subplots(n_show, 4, figsize=(20, 4.0 * n_show))
    fig.patch.set_facecolor("#0F172A")
    plt.rcParams["text.color"] = "#E2E8F0"

    if n_show == 1:
        axes = axes.reshape(1, -1)

    col_headers = [
        "1. Raw Input",
        "2. Enhanced (Stage 1)",
        "3. Binary Mask (Stage 2)",
        "4. Detection vs GT",
    ]

    for r in range(n_show):
        item = results[r]
        raw_rgb = cv2.cvtColor(item["raw_image"], cv2.COLOR_BGR2RGB)
        enh_rgb = cv2.cvtColor(item["enhanced_image"], cv2.COLOR_BGR2RGB)
        mask = item["binary_mask"]
        eval_overlay_rgb = cv2.cvtColor(item["eval_overlay"], cv2.COLOR_BGR2RGB)
        ev = item["eval"]

        axes[r, 0].imshow(raw_rgb)
        axes[r, 0].set_title(f"[{r+1}] {item['name']}", fontsize=9, color="white")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(enh_rgb)
        axes[r, 1].set_title("Stage 1 Enhanced", fontsize=9, color="#38BDF8")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(mask, cmap="gray")
        axes[r, 2].set_title("Verified Binary Mask", fontsize=9, color="#A78BFA")
        axes[r, 2].axis("off")

        axes[r, 3].imshow(eval_overlay_rgb)
        stats = f"TP={ev['tp']} FP={ev['fp']} FN={ev['fn']}"
        axes[r, 3].set_title(f"Detection Eval\n{stats}", fontsize=9, color="#34D399")
        axes[r, 3].axis("off")

    for c in range(4):
        axes[0, c].text(
            0.5, 1.22, col_headers[c],
            transform=axes[0, c].transAxes,
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#F8FAFC"
        )

    plt.suptitle(
        "GTSDB 150-IMAGE BATCH EVALUATION: DETECTION vs GROUND TRUTH",
        fontsize=14, fontweight="bold", color="#38BDF8", y=1.002
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def main():
    root = Path(__file__).resolve().parent
    gtsdb_dir = root / "gtsdb_dataset"

    print("\n" + "=" * 70)
    print("GTSDB 150-IMAGE BATCH EVALUATION")
    print("Stage 1: Image Enhancement | Stage 2: Color Segmentation")
    print("Strictly Image Processing — NO CNN, NO YOLO")
    print("=" * 70)

    # Select 150 images
    image_pairs = select_150_images(gtsdb_dir)
    print(f"\nSelected {len(image_pairs)} images for evaluation\n")

    # Prepare output directories
    out_dir = root / "outputs" / "gtsdb_150_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_dir = out_dir / "per_image"
    per_image_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig()

    # Accumulators
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_results = []
    csv_rows = []

    for idx, (img_path, label_path) in enumerate(image_pairs):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{idx+1}] SKIP: Could not load {img_path.name}")
            continue

        img_h, img_w = img.shape[:2]
        scene_name = img_path.stem

        # Load ground truth
        gt_bboxes = load_ground_truth(label_path, img_w, img_h)

        # Run pipeline
        res = run_full_pipeline(img, config)
        s1 = res["stage1"]
        s2 = res["stage2"]

        detected_bboxes = s2["bounding_boxes"]

        # Evaluate against ground truth
        eval_result = evaluate_detections(detected_bboxes, gt_bboxes, iou_threshold=0.25)

        total_tp += eval_result["tp"]
        total_fp += eval_result["fp"]
        total_fn += eval_result["fn"]

        # Draw evaluation overlay
        eval_overlay = draw_evaluation_overlay(
            s1["enhanced_image"], detected_bboxes, gt_bboxes, eval_result
        )

        # Save per-image outputs
        cv2.imwrite(str(per_image_dir / f"{scene_name}_eval.png"), eval_overlay)
        cv2.imwrite(str(per_image_dir / f"{scene_name}_mask.png"), s2["binary_mask_cleaned"])

        # Collect for gallery
        all_results.append({
            "name": scene_name[:30],
            "raw_image": img,
            "enhanced_image": s1["enhanced_image"],
            "binary_mask": s2["binary_mask_cleaned"],
            "eval_overlay": eval_overlay,
            "eval": eval_result,
        })

        # CSV row
        csv_rows.append({
            "image": img_path.name,
            "gt_signs": len(gt_bboxes),
            "detected": len(detected_bboxes),
            "tp": eval_result["tp"],
            "fp": eval_result["fp"],
            "fn": eval_result["fn"],
            "precision": f"{eval_result['precision']:.3f}",
            "recall": f"{eval_result['recall']:.3f}",
            "f1": f"{eval_result['f1']:.3f}",
        })

        status = "OK" if eval_result["fp"] == 0 and eval_result["fn"] == 0 else "  "
        print(
            f"  [{idx+1:3d}/{len(image_pairs)}] {status} {scene_name[:35]:35s} "
            f"GT={len(gt_bboxes)} Det={len(detected_bboxes)} "
            f"TP={eval_result['tp']} FP={eval_result['fp']} FN={eval_result['fn']}"
        )

    # Compute overall metrics
    overall_precision = total_tp / max(1, total_tp + total_fp)
    overall_recall = total_tp / max(1, total_tp + total_fn)
    overall_f1 = 2 * overall_precision * overall_recall / max(1e-8, overall_precision + overall_recall)

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"  Total Images:     {len(all_results)}")
    print(f"  Total GT Signs:   {total_tp + total_fn}")
    print(f"  Total Detections: {total_tp + total_fp}")
    print(f"  True Positives:   {total_tp}")
    print(f"  False Positives:  {total_fp}")
    print(f"  False Negatives:  {total_fn}")
    print(f"  Precision:        {overall_precision:.4f}")
    print(f"  Recall:           {overall_recall:.4f}")
    print(f"  F1 Score:         {overall_f1:.4f}")
    print("=" * 70)

    # Save CSV report
    csv_path = out_dir / "evaluation_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    # Save summary stats
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("GTSDB 150-Image Batch Evaluation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Images:     {len(all_results)}\n")
        f.write(f"Total GT Signs:   {total_tp + total_fn}\n")
        f.write(f"Total Detections: {total_tp + total_fp}\n")
        f.write(f"True Positives:   {total_tp}\n")
        f.write(f"False Positives:  {total_fp}\n")
        f.write(f"False Negatives:  {total_fn}\n")
        f.write(f"Precision:        {overall_precision:.4f}\n")
        f.write(f"Recall:           {overall_recall:.4f}\n")
        f.write(f"F1 Score:         {overall_f1:.4f}\n")

    # Generate master gallery (show first 20 images)
    print(f"\n[+] Generating master gallery grid...")
    gallery_path = out_dir / "master_gallery.png"
    generate_summary_gallery(all_results, gallery_path, max_rows=20)

    print(f"\n[OK] Batch evaluation completed!")
    print(f"  - Per-image results: {per_image_dir}")
    print(f"  - CSV report:        {csv_path}")
    print(f"  - Summary:           {summary_path}")
    print(f"  - Master gallery:    {gallery_path}\n")


if __name__ == "__main__":
    main()
