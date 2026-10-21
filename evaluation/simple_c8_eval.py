"""Simple C8 Evaluation with Visual Outputs

This is a simplified evaluator that generates visual overlays
showing detected traffic signs with ground truth boxes.

Usage:
    python evaluation/simple_c8_eval.py --images 00000.ppm,00001.ppm
    python evaluation/simple_c8_eval.py --validation
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig, DatasetConfig
from src.pipeline import run_stage1
from src.stage2_segmentation.thresholding import segment_traffic_sign_colors
from src.stage2_segmentation.morphology import apply_morphology, non_max_suppression_boxes


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
    
    box1_x1, box1_y1, box1_x2, box1_y2 = x1, y1, x1 + w1, y1 + h1
    box2_x1, box2_y1, box2_x2, box2_y2 = x2, y2, x2 + w2, y2 + h2
    
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_one_to_one(det_boxes, gt_boxes, iou_threshold=0.5):
    """Greedy one-to-one matching."""
    if not det_boxes or not gt_boxes:
        return {'tp': 0, 'fp': len(det_boxes), 'fn': len(gt_boxes), 'matches': []}
    
    # Compute all IoUs
    iou_matrix = []
    for i, det in enumerate(det_boxes):
        row = []
        for j, gt in enumerate(gt_boxes):
            gt_xywh = (gt['left'], gt['top'], gt['right'] - gt['left'], gt['bottom'] - gt['top'])
            iou = compute_iou_xywh(det, gt_xywh)
            row.append((iou, i, j))
        iou_matrix.append(row)
    
    # Filter by threshold and sort
    valid_pairs = []
    for row in iou_matrix:
        for iou, det_idx, gt_idx in row:
            if iou >= iou_threshold:
                valid_pairs.append((iou, det_idx, gt_idx))
    
    valid_pairs.sort(reverse=True, key=lambda x: x[0])
    
    # Greedy matching
    matched_det = set()
    matched_gt = set()
    matches = []
    
    for iou, det_idx, gt_idx in valid_pairs:
        if det_idx not in matched_det and gt_idx not in matched_gt:
            matched_det.add(det_idx)
            matched_gt.add(gt_idx)
            matches.append({
                'det_idx': det_idx,
                'gt_idx': gt_idx,
                'iou': iou,
                'class_id': gt_boxes[gt_idx]['class_id']
            })
    
    tp = len(matches)
    fp = len(det_boxes) - tp
    fn = len(gt_boxes) - tp
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'matches': matches,
        'precision': tp / max(1, tp + fp),
        'recall': tp / max(1, tp + fn),
        'f1': 2 * tp / max(1e-8, 2 * tp + fp + fn)
    }


def extract_hog_features_from_crop(crop_bgr, hog_config):
    """Extract HOG features from a crop."""
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    crop_resized = cv2.resize(crop_gray, tuple(hog_config['window_size']), interpolation=cv2.INTER_LINEAR)
    
    hog = cv2.HOGDescriptor(
        _winSize=tuple(hog_config['window_size']),
        _blockSize=(hog_config['pixels_per_cell'][0] * hog_config['cells_per_block'][0],
                   hog_config['pixels_per_cell'][1] * hog_config['cells_per_block'][1]),
        _blockStride=tuple(hog_config['pixels_per_cell']),
        _cellSize=tuple(hog_config['pixels_per_cell']),
        _nbins=hog_config['orientations'],
    )
    
    features = hog.compute(crop_resized).flatten()
    return features


def simple_detection(image_bgr, config):
    """Simple color-based detection (Stage 2 only, no M3_D filters)."""
    # Stage 1 enhancement
    stage1 = run_stage1(image_bgr, config.stage1)
    enhanced = stage1['enhanced_image']
    
    # Stage 2 segmentation
    color_masks = segment_traffic_sign_colors(enhanced, config.stage2)
    
    # Morphology
    cleaned_red = apply_morphology(color_masks['mask_red'], 3, 3, 1)
    cleaned_blue = apply_morphology(color_masks['mask_blue'], 3, 3, 1)
    cleaned_yellow = apply_morphology(color_masks['mask_yellow'], 3, 3, 1)
    
    # Find contours in each color
    candidates = []
    h, w = enhanced.shape[:2]
    
    for color_name, mask in [('red', cleaned_red), ('blue', cleaned_blue), ('yellow', cleaned_yellow)]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 40 or area > 18000:
                continue
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 8 or bh < 8 or bw > 180 or bh > 180:
                continue
            
            ar = bw / max(1, bh)
            if ar < 0.35 or ar > 2.0:
                continue
            
            candidates.append({
                'x': x,
                'y': y,
                'w': bw,
                'h': bh,
                'color': color_name,
                'area': area
            })
    
    # NMS
    if candidates:
        boxes = [(c['x'], c['y'], c['w'], c['h']) for c in candidates]
        nms_boxes = non_max_suppression_boxes(boxes, iou_threshold=0.40)
        candidates = [c for c in candidates if (c['x'], c['y'], c['w'], c['h']) in nms_boxes]
    
    return candidates


def main():
    parser = argparse.ArgumentParser(description='Simple C8 Evaluation with Visual Outputs')
    parser.add_argument('--images', type=str, help='Comma-separated list of image filenames')
    parser.add_argument('--validation', action='store_true', help='Use 98 validation images')
    
    args = parser.parse_args()
    
    # Determine image list
    if args.validation:
        manifest_path = Path('data/split_manifest.csv')
        sample_images = []
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == 'val':
                    sample_images.append(row['image_name'])
        sample_images = sorted(sample_images)
        print(f"Using validation images: {len(sample_images)}")
    elif args.images:
        sample_images = [img.strip() for img in args.images.split(',')]
        print(f"Using provided images: {len(sample_images)}")
    else:
        print("ERROR: Must specify --images or --validation")
        return
    
    # Load GT annotations
    root = Path('.').resolve()
    ds_cfg = DatasetConfig()
    gtsdb_dir = root / ds_cfg.gtsdb_dir
    gt_file = root / ds_cfg.gtsdb_gt_file
    all_gt = load_gtsdb_annotations(gt_file)
    
    gt_count = sum(len(all_gt.get(img, [])) for img in sample_images)
    print(f"Total GT boxes: {gt_count}")
    
    # C8 configuration
    cfg = PipelineConfig()
    cfg.stage2.independent_color_morphology = True
    cfg.stage2.morph_kernel_size = 3
    cfg.stage2.morph_close_iterations = 3
    cfg.stage2.morph_open_iterations = 1
    cfg.stage2.nms_iou_threshold = 0.40
    cfg.stage2.lab_red_lower = (20, 130, 100)
    cfg.stage2.lab_blue_lower = (20, 60, 20)
    cfg.stage2.lab_blue_upper = (230, 145, 155)
    cfg.stage2.lab_yellow_lower = (40, 100, 140)
    
    # Create output directory
    output_dir = Path('results/detection_output/simple_eval')
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / 'visual_overlays'
    overlays_dir.mkdir(exist_ok=True)
    
    # Process images
    all_detections = []
    total_tp = total_fp = total_fn = 0
    
    print("\nProcessing images...")
    for img_name in sample_images:
        img_path = gtsdb_dir / img_name
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  Warning: Could not load {img_name}")
            continue
        
        # Detect
        candidates = simple_detection(img_bgr, cfg)
        det_boxes = [(int(c['x']), int(c['y']), int(c['w']), int(c['h'])) for c in candidates]
        
        # Evaluate
        gt_boxes = all_gt.get(img_name, [])
        ev = evaluate_one_to_one(det_boxes, gt_boxes, 0.50)
        
        total_tp += ev['tp']
        total_fp += ev['fp']
        total_fn += ev['fn']
        
        # Store detections
        for det in candidates:
            all_detections.append({
                'image': img_name,
                'x': det['x'],
                'y': det['y'],
                'w': det['w'],
                'h': det['h'],
                'color': det['color']
            })
        
        # Draw overlay
        overlay = img_bgr.copy()
        
        # Draw GT boxes (green)
        for gt in gt_boxes:
            x, y, w, h = gt['left'], gt['top'], gt['right'] - gt['left'], gt['bottom'] - gt['top']
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(overlay, "GT", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw detections
        for det in det_boxes:
            x, y, w, h = det
            is_matched = False
            for match in ev['matches']:
                if match['det_idx'] == det_boxes.index(det):
                    is_matched = True
                    break
            
            color = (255, 0, 0) if is_matched else (0, 0, 255)
            label = "TP" if is_matched else "FP"
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            cv2.putText(overlay, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save overlay
        overlay_path = overlays_dir / img_name.replace('.ppm', '.png')
        cv2.imwrite(str(overlay_path), overlay)
        
        print(f"  {img_name}: {len(det_boxes)} detections, TP={ev['tp']}, FP={ev['fp']}, FN={ev['fn']}")
    
    # Metrics
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * total_tp / max(1e-8, 2 * total_tp + total_fp + total_fn)
    
    print(f"\nOverall Metrics:")
    print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  F1: {f1:.2%}")
    
    # Save metrics
    metrics = {
        'images_evaluated': len(sample_images),
        'gt_boxes': gt_count,
        'total_detections': len(all_detections),
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Visual overlays: {overlays_dir}")


if __name__ == '__main__':
    main()
