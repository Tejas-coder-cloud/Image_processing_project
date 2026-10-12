"""Investigation script for Stage 3 candidate detection.

This script:
1. Loads GTSDB images with ground truth
2. Runs the existing pipeline unchanged
3. Compares detections with ground truth
4. Saves detailed analysis outputs
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import csv

from src.config import PipelineConfig
from src.pipeline import run_full_pipeline


def load_gtsdb_gt(gt_path: Path, img_w: int, img_h: int, img_name: str) -> List[Dict]:
    """Load GTSDB ground truth format: filename;left;top;right;bottom;class_id

    Filters to only return GT entries for the specified image name.
    """
    gt_list = []
    if not gt_path.exists():
        return gt_list

    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 6:
                continue
            # Format: filename;left;top;right;bottom;class_id
            filename = parts[0]
            if filename != img_name:
                continue

            left = int(parts[1])
            top = int(parts[2])
            right = int(parts[3])
            bottom = int(parts[4])
            class_id = int(parts[5])

            gt_list.append({
                "class_id": class_id,
                "x": left,
                "y": top,
                "w": right - left,
                "h": bottom - top,
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
    gt_bboxes: List[Dict],
    iou_threshold: float = 0.25,
) -> Dict:
    """Match detected bboxes to ground truth using IoU."""
    n_gt = len(gt_bboxes)
    n_det = len(detected_bboxes)

    matched_gt = set()
    matched_det = set()
    matches = []

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
                "class_id": gt["class_id"],
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


def draw_analysis_overlay(
    image_bgr: np.ndarray,
    detected_bboxes: List[Tuple[int, int, int, int]],
    gt_bboxes: List[Dict],
    eval_result: Dict,
) -> np.ndarray:
    """Draw annotated overlay showing TP, FP, and FN."""
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

    # Draw missed ground truth signs in CYAN (dashed)
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
        cv2.putText(overlay, label, (x, max(12, y - 4)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

    # Draw stats bar
    stats_text = f"TP={eval_result['tp']}  FP={eval_result['fp']}  FN={eval_result['fn']}  P={eval_result['precision']:.2f}  R={eval_result['recall']:.2f}"
    bar_y = h - 25
    cv2.rectangle(overlay, (0, bar_y - 5), (w, h), (0, 0, 0), -1)
    cv2.putText(overlay, stats_text, (8, bar_y + 12),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def main():
    root = Path(__file__).resolve().parent
    gtsdb_dir = root / "data" / "TrainIJCNN2013" / "TrainIJCNN2013"
    gt_file = gtsdb_dir / "gt.txt"
    output_dir = root / "investigation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select a diverse sample of 20 images
    test_images = [
        "00000.ppm", "00001.ppm", "00002.ppm", "00003.ppm", "00004.ppm",
        "00005.ppm", "00006.ppm", "00007.ppm", "00008.ppm", "00009.ppm",
        "00010.ppm", "00011.ppm", "00012.ppm", "00013.ppm", "00014.ppm",
        "00015.ppm", "00016.ppm", "00017.ppm", "00018.ppm", "00019.ppm",
    ]

    config = PipelineConfig()

    results = []
    csv_rows = []

    print("\n" + "=" * 70)
    print("STAGE 3 INVESTIGATION: RUNNING PIPELINE ON GTSDB SAMPLE")
    print("=" * 70 + "\n")

    for img_name in test_images:
        img_path = gtsdb_dir / img_name
        if not img_path.exists():
            print(f"  SKIP: {img_name} not found")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  SKIP: Could not load {img_name}")
            continue

        img_h, img_w = img.shape[:2]

        # Load ground truth for this image
        img_gt = load_gtsdb_gt(gt_file, img_w, img_h, img_name)

        # Run pipeline
        res = run_full_pipeline(img, config)
        s1 = res["stage1"]
        s2 = res["stage2"]

        detected_bboxes = s2["bounding_boxes"]

        # Evaluate
        eval_result = evaluate_detections(detected_bboxes, img_gt, iou_threshold=0.25)

        # Draw overlay
        overlay = draw_analysis_overlay(s1["enhanced_image"], detected_bboxes, img_gt, eval_result)

        # Save outputs
        stem = img_path.stem
        cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), overlay)
        cv2.imwrite(str(output_dir / f"{stem}_mask.png"), s2["binary_mask_cleaned"])
        cv2.imwrite(str(output_dir / f"{stem}_enhanced.png"), s1["enhanced_image"])

        # Save detailed detection info
        with open(output_dir / f"{stem}_details.txt", "w") as f:
            f.write(f"Image: {img_name}\n")
            f.write(f"Ground Truth Signs: {len(img_gt)}\n")
            f.write(f"Detected Candidates: {len(detected_bboxes)}\n")
            f.write(f"True Positives: {eval_result['tp']}\n")
            f.write(f"False Positives: {eval_result['fp']}\n")
            f.write(f"False Negatives: {eval_result['fn']}\n")
            f.write(f"Precision: {eval_result['precision']:.3f}\n")
            f.write(f"Recall: {eval_result['recall']:.3f}\n")
            f.write(f"F1: {eval_result['f1']:.3f}\n\n")

            f.write("GROUND TRUTH BOXES:\n")
            for i, gt in enumerate(img_gt):
                f.write(f"  GT #{i}: class={gt['class_id']}, x={gt['x']}, y={gt['y']}, w={gt['w']}, h={gt['h']}\n")

            f.write("\nDETECTED BOXES:\n")
            for i, box in enumerate(detected_bboxes):
                f.write(f"  Det #{i}: x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}\n")

            f.write("\nMATCHES (TP):\n")
            for match in eval_result["matches"]:
                f.write(f"  GT #{match['gt_idx']} <-> Det #{match['det_idx']}: IoU={match['iou']:.3f}, class={match['class_id']}\n")

            f.write("\nFALSE POSITIVES:\n")
            for di in eval_result["unmatched_det_indices"]:
                box = detected_bboxes[di]
                # Find best matching GT for each FP
                best_iou = 0.0
                best_gi = -1
                for gi, gt in enumerate(img_gt):
                    iou = compute_iou(box, (gt["x"], gt["y"], gt["w"], gt["h"]))
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi
                f.write(f"  Det #{di}: x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}")
                if best_gi >= 0:
                    f.write(f" (best IoU with GT #{best_gi}: {best_iou:.3f})\n")
                else:
                    f.write("\n")

            f.write("\nFALSE NEGATIVES (MISSED):\n")
            for gi in eval_result["unmatched_gt_indices"]:
                f.write(f"  GT #{gi}: class={img_gt[gi]['class_id']}, {img_gt[gi]}\n")

        results.append({
            "name": img_name,
            "img": img,
            "overlay": overlay,
            "mask": s2["binary_mask_cleaned"],
            "eval": eval_result,
        })

        csv_rows.append({
            "image": img_name,
            "gt_signs": len(img_gt),
            "detected": len(detected_bboxes),
            "tp": eval_result["tp"],
            "fp": eval_result["fp"],
            "fn": eval_result["fn"],
            "precision": f"{eval_result['precision']:.3f}",
            "recall": f"{eval_result['recall']:.3f}",
            "f1": f"{eval_result['f1']:.3f}",
        })

        status = "OK" if eval_result["fp"] == 0 and eval_result["fn"] == 0 else "  "
        print(f"  {status} {img_name:15s} GT={len(img_gt):2d} Det={len(detected_bboxes):2d} "
              f"TP={eval_result['tp']:2d} FP={eval_result['fp']:2d} FN={eval_result['fn']:2d} "
              f"P={eval_result['precision']:.2f} R={eval_result['recall']:.2f}")

    # Save CSV summary
    csv_path = output_dir / "investigation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    # Overall statistics
    total_tp = sum(r["eval"]["tp"] for r in results)
    total_fp = sum(r["eval"]["fp"] for r in results)
    total_fn = sum(r["eval"]["fn"] for r in results)
    overall_p = total_tp / max(1, total_tp + total_fp)
    overall_r = total_tp / max(1, total_tp + total_fn)
    overall_f1 = 2 * overall_p * overall_r / max(1e-8, overall_p + overall_r)

    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    print(f"  Total Images:     {len(results)}")
    print(f"  Total GT Signs:   {total_tp + total_fn}")
    print(f"  Total Detections: {total_tp + total_fp}")
    print(f"  True Positives:   {total_tp}")
    print(f"  False Positives:  {total_fp}")
    print(f"  False Negatives:  {total_fn}")
    print(f"  Precision:        {overall_p:.4f}")
    print(f"  Recall:           {overall_r:.4f}")
    print(f"  F1 Score:         {overall_f1:.4f}")
    print("=" * 70)

    print(f"\n[OK] Investigation complete!")
    print(f"  - Outputs saved to: {output_dir}")
    print(f"  - CSV summary: {csv_path}\n")


if __name__ == "__main__":
    main()
