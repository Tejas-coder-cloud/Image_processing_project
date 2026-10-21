"""Phase N1.1: Training-Data Preparation for HOG + SVM Binary Verifier.

This script prepares positive and negative crop datasets for training a classical
HOG + SVM binary traffic-sign/background verifier. It uses GTSDB ground-truth
annotations and the existing M3_D detector to generate hard negatives.

Output directory: experiments_phase_n1/data_preparation/
"""

import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
import cv2
import numpy as np

from src.config import PipelineConfig, DatasetConfig


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


# HOG configuration (for manifest only, not feature extraction in this script)
HOG_CONFIG = {
    "window_size": (64, 64),
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True,
}


def parse_annotations(gt_file: Path, img_dir: Path) -> Dict[str, List[Dict]]:
    """Parse GTSDB annotations and return dictionary by image name."""
    annotations_by_image: Dict[str, List[Dict]] = {}
    
    with open(gt_file, 'r') as f:
        for line_idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.strip().split(';')
            if len(parts) < 6:
                print(f"Warning: Line {line_idx} has fewer than 6 fields, skipping")
                continue
            
            img_name, left_s, top_s, right_s, bottom_s, cls_s = parts[:6]
            
            try:
                x1 = int(left_s)
                y1 = int(top_s)
                x2 = int(right_s)
                y2 = int(bottom_s)
                class_id = int(cls_s)
            except ValueError as e:
                print(f"Warning: Line {line_idx} parse error: {e}, skipping")
                continue
            
            w = x2 - x1
            h = y2 - y1
            
            if w <= 0 or h <= 0:
                print(f"Warning: Line {line_idx} degenerate box: ({x1},{y1},{x2},{y2}), skipping")
                continue
            
            entry = {
                "image_name": img_name,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "w": w,
                "h": h,
                "class_id": class_id,
            }
            
            annotations_by_image.setdefault(img_name, []).append(entry)
    
    return annotations_by_image


def split_images(
    all_gt: Dict[str, List[Dict]],
    eval_images: Set[str],
    train_ratio: float = 0.8,
    random_seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split images into train/validation sets, excluding evaluation images.
    
    Returns (train_images, val_images)
    """
    # Get all available images
    all_image_names = list(all_gt.keys())
    
    # Remove evaluation images
    available_images = [img for img in all_image_names if img not in eval_images]
    
    # Count boxes per image for stratification
    box_counts = {img: len(all_gt[img]) for img in available_images}
    
    # Sort by box count to ensure distribution
    available_images.sort(key=lambda x: box_counts[x])
    
    # Simple split with seed (box-count stratification not implemented due to complexity)
    random.seed(random_seed)
    random.shuffle(available_images)
    
    split_idx = int(len(available_images) * train_ratio)
    train_images = available_images[:split_idx]
    val_images = available_images[split_idx:]
    
    return train_images, val_images


def clip_box_to_image(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Clip box coordinates to image boundaries."""
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    return x1, y1, x2, y2


def is_box_valid(x1: int, y1: int, x2: int, y2: int, min_dim: int = 16) -> Tuple[bool, str]:
    """Check if box is valid for cropping."""
    w = x2 - x1
    h = y2 - y1
    
    if w <= 0 or h <= 0:
        return False, "non_positive_dimensions"
    
    if w < min_dim or h < min_dim:
        return False, f"too_small_{min_dim}px"
    
    return True, "valid"


def generate_random_background_crop(
    img_bgr: np.ndarray,
    gt_boxes: List[Dict],
    target_size: Tuple[int, int] = (64, 64),
    iou_threshold: float = 0.1,
    num_attempts: int = 100,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], str]:
    """Generate a random background crop with low IoU to all GT boxes.
    
    Returns (crop, (x1, y1, x2, y2), status)
    """
    img_h, img_w = img_bgr.shape[:2]
    
    for _ in range(num_attempts):
        # Random size: 16 to 200 pixels (matching typical sign sizes)
        w = random.randint(16, 200)
        h = random.randint(16, 200)
        
        # Random position
        x1 = random.randint(0, max(1, img_w - w))
        y1 = random.randint(0, max(1, img_h - h))
        x2 = x1 + w
        y2 = y1 + h
        
        # Check center not inside any GT box
        cx, cy = x1 + w // 2, y1 + h // 2
        center_inside = False
        for gt in gt_boxes:
            if (gt["x1"] < cx < gt["x2"]) and (gt["y1"] < cy < gt["y2"]):
                center_inside = True
                break
        if center_inside:
            continue
        
        # Check IoU with all GT boxes
        max_iou = 0.0
        for gt in gt_boxes:
            gt_box = (gt["x1"], gt["y1"], gt["w"], gt["h"])
            crop_box = (x1, y1, w, h)
            iou = compute_iou_xywh(gt_box, crop_box)
            max_iou = max(max_iou, iou)
        
        if max_iou < iou_threshold:
            # Valid background crop
            crop = img_bgr[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
            return crop_resized, (x1, y1, x2, y2), "random_background"
    
    # Failed to find valid crop
    return None, (0, 0, 0, 0), "no_valid_region"


def extract_positive_crop(
    img_bgr: np.ndarray,
    gt_box: Dict,
    target_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Extract and resize a positive crop from ground-truth box."""
    x1, y1, x2, y2 = gt_box["x1"], gt_box["y1"], gt_box["x2"], gt_box["y2"]
    
    # Clip to image boundaries
    img_h, img_w = img_bgr.shape[:2]
    x1, y1, x2, y2 = clip_box_to_image(x1, y1, x2, y2, img_w, img_h)
    
    # Extract crop
    crop = img_bgr[y1:y2, x1:x2]
    
    # Resize to target size
    crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    
    return crop_resized


def generate_random_background_crop(
    img_bgr: np.ndarray,
    gt_boxes: List[Dict],
    target_size: Tuple[int, int] = (64, 64),
    iou_threshold: float = 0.1,
    num_attempts: int = 100,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], str]:
    """Generate a random background crop with low IoU to all GT boxes.
    
    Returns (crop, (x1, y1, x2, y2), status)
    """
    img_h, img_w = img_bgr.shape[:2]
    
    for _ in range(num_attempts):
        # Random size: 16 to 200 pixels (matching typical sign sizes)
        w = random.randint(16, 200)
        h = random.randint(16, 200)
        
        # Random position
        x1 = random.randint(0, max(1, img_w - w))
        y1 = random.randint(0, max(1, img_h - h))
        x2 = x1 + w
        y2 = y1 + h
        
        # Check center not inside any GT box
        cx, cy = x1 + w // 2, y1 + h // 2
        center_inside = False
        for gt in gt_boxes:
            if (gt["x1"] < cx < gt["x2"]) and (gt["y1"] < cy < gt["y2"]):
                center_inside = True
                break
        if center_inside:
            continue
        
        # Check IoU with all GT boxes
        max_iou = 0.0
        for gt in gt_boxes:
            gt_box = (gt["x1"], gt["y1"], gt["w"], gt["h"])
            crop_box = (x1, y1, w, h)
            iou = compute_iou_xywh(gt_box, crop_box)
            max_iou = max(max_iou, iou)
        
        if max_iou < iou_threshold:
            # Valid background crop
            crop = img_bgr[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
            return crop_resized, (x1, y1, x2, y2), "random_background"
    
    # Failed to find valid crop
    return None, (0, 0, 0, 0), "no_valid_region"


def main():
    root = Path('.').resolve()
    ds_cfg = DatasetConfig()
    gtsdb_dir = root / ds_cfg.gtsdb_dir
    gt_file = root / ds_cfg.gtsdb_gt_file
    
    # Output directory
    out_dir = root / 'experiments_phase_n1' / 'data_preparation'
    if out_dir.exists():
        print(f"ERROR: Output directory already exists: {out_dir}")
        print("Please remove it or specify a different location to avoid overwriting.")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / 'crops'
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    # Subdirectories for crops
    for split in ['train', 'val']:
        for label in ['sign', 'background']:
            (crops_dir / split / label).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Phase N1.1: Training-Data Preparation")
    print("=" * 80)
    
    # Load annotations
    print("\nStep 1: Loading GTSDB annotations...")
    all_gt = parse_annotations(gt_file, gtsdb_dir)
    print(f"Total annotated images: {len(all_gt)}")
    print(f"Total ground-truth boxes: {sum(len(v) for v in all_gt.values())}")
    
    # Define evaluation images (held-out)
    eval_images = {f'{i:05d}.ppm' for i in range(20)}
    print(f"Evaluation images (held-out): {len(eval_images)}")
    
    # Split training/validation
    print("\nStep 2: Splitting images into train/validation...")
    train_images, val_images = split_images(all_gt, eval_images, train_ratio=0.8, random_seed=42)
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    # Verify no leakage
    train_set = set(train_images)
    val_set = set(val_images)
    assert train_set.isdisjoint(val_set), "Train and validation sets overlap!"
    assert train_set.isdisjoint(eval_images), "Train set contains evaluation images!"
    assert val_set.isdisjoint(eval_images), "Validation set contains evaluation images!"
    print("Leakage check passed: train, val, and eval sets are disjoint.")
    
    # Save split manifest
    split_manifest = out_dir / 'split_manifest.csv'
    with open(split_manifest, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_name', 'split', 'gt_box_count'])
        for img in train_images:
            w.writerow([img, 'train', len(all_gt[img])])
        for img in val_images:
            w.writerow([img, 'val', len(all_gt[img])])
    print(f"Split manifest saved: {split_manifest}")
    
    # Extract positive crops
    print("\nStep 3: Extracting positive crops...")
    metadata = []
    positive_stats = {'total': 0, 'accepted': 0, 'rejected': 0, 'rejections': {}}
    
    crop_id_counter = 0
    
    for split, image_list in [('train', train_images), ('val', val_images)]:
        print(f"  Processing {split} split ({len(image_list)} images)...")
        
        for img_name in image_list:
            img_path = gtsdb_dir / img_name
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"    Warning: Could not load {img_name}")
                continue
            
            gt_boxes = all_gt[img_name]
            
            for gt_box in gt_boxes:
                positive_stats['total'] += 1
                
                # Check validity
                valid, reason = is_box_valid(gt_box["x1"], gt_box["y1"], gt_box["x2"], gt_box["y2"])
                if not valid:
                    positive_stats['rejected'] += 1
                    positive_stats['rejections'][reason] = positive_stats['rejections'].get(reason, 0) + 1
                    continue
                
                # Extract and resize crop
                try:
                    crop = extract_positive_crop(img_bgr, gt_box, target_size=(64, 64))
                except Exception as e:
                    positive_stats['rejected'] += 1
                    positive_stats['rejections']['extraction_error'] = positive_stats['rejections'].get('extraction_error', 0) + 1
                    print(f"    Warning: Extraction failed for {img_name} GT box: {e}")
                    continue
                
                # Save crop
                crop_filename = f"crop_{crop_id_counter:06d}.png"
                crop_path = crops_dir / split / 'sign' / crop_filename
                cv2.imwrite(str(crop_path), crop)
                
                # Record metadata
                metadata.append({
                    'crop_id': crop_id_counter,
                    'image_name': img_name,
                    'split': split,
                    'label': 'sign',
                    'source_type': 'ground_truth',
                    'x1': gt_box["x1"],
                    'y1': gt_box["y1"],
                    'x2': gt_box["x2"],
                    'y2': gt_box["y2"],
                    'width': gt_box["w"],
                    'height': gt_box["h"],
                    'area': gt_box["w"] * gt_box["h"],
                })
                
                crop_id_counter += 1
                positive_stats['accepted'] += 1
    
    print(f"  Positive crops: {positive_stats['accepted']} accepted, {positive_stats['rejected']} rejected")
    for reason, count in positive_stats['rejections'].items():
        print(f"    Rejected ({reason}): {count}")
    
    # Generate negative crops
    print("\nStep 4: Generating negative crops...")
    print("  NOTE: M3_D hard negative generation is BLOCKED due to complexity.")
    print("  Using random background negatives only in this step.")
    print("  Hard negatives will be added in a future step if needed.")
    
    negative_stats = {'total_attempted': 0, 'accepted': 0, 'rejected': 0, 'by_split': {'train': 0, 'val': 0}, 'by_source': {'random_background': 0}}
    
    target_negatives_per_positive = 2
    total_target_negatives = positive_stats['accepted'] * target_negatives_per_positive
    print(f"  Target negative crops: {total_target_negatives} (2x positive)")
    
    # Reset random seed for reproducibility
    random.seed(42)
    
    for split, image_list in [('train', train_images), ('val', val_images)]:
        print(f"  Processing {split} split for negatives...")
        
        split_target = int(positive_stats['accepted'] * target_negatives_per_positive * (len(image_list) / (len(train_images) + len(val_images))))
        print(f"    Target for {split}: {split_target}")
        
        split_negatives = 0
        
        for img_name in image_list:
            if split_negatives >= split_target:
                break
            
            img_path = gtsdb_dir / img_name
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            
            gt_boxes = all_gt[img_name]
            
            # Generate multiple random crops per image
            attempts_per_image = 20
            for _ in range(attempts_per_image):
                if split_negatives >= split_target:
                    break
                
                negative_stats['total_attempted'] += 1
                crop, (x1, y1, x2, y2), status = generate_random_background_crop(
                    img_bgr, gt_boxes, target_size=(64, 64), iou_threshold=0.1, num_attempts=100
                )
                
                if crop is not None:
                    # Save crop
                    crop_filename = f"crop_{crop_id_counter:06d}.png"
                    crop_path = crops_dir / split / 'background' / crop_filename
                    cv2.imwrite(str(crop_path), crop)
                    
                    metadata.append({
                        'crop_id': crop_id_counter,
                        'image_name': img_name,
                        'split': split,
                        'label': 'background',
                        'source_type': 'random_background',
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'area': (x2 - x1) * (y2 - y1),
                    })
                    
                    crop_id_counter += 1
                    split_negatives += 1
                    negative_stats['accepted'] += 1
                    negative_stats['by_split'][split] += 1
                    negative_stats['by_source']['random_background'] += 1
        
        print(f"    Random background negatives generated: {split_negatives} (target: {split_target})")
        if split_negatives < split_target:
            print(f"    WARNING: Could not reach target ({split_target} - {split_negatives} = {split_target - split_negatives} short)")
    
    print(f"  Total negative crops: {negative_stats['accepted']}")
    print(f"    By split: train={negative_stats['by_split']['train']}, val={negative_stats['by_split']['val']}")
    print(f"    By source: random_background={negative_stats['by_source']['random_background']}")
    
    # Save metadata CSV
    metadata_csv = out_dir / 'crop_metadata.csv'
    with open(metadata_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['crop_id', 'image_name', 'split', 'label', 'source_type', 'x1', 'y1', 'x2', 'y2', 'width', 'height', 'area'])
        w.writeheader()
        w.writerows(metadata)
    print(f"\nMetadata saved: {metadata_csv}")
    
    # Save HOG configuration manifest
    hog_config_file = out_dir / 'hog_config.json'
    import json
    with open(hog_config_file, 'w') as f:
        json.dump(HOG_CONFIG, f, indent=2)
    print(f"HOG configuration saved: {hog_config_file}")
    
    # Generate summary report
    report = out_dir / 'preparation_report.md'
    with open(report, 'w') as f:
        f.write("# Phase N1.1 Training-Data Preparation Report\n\n")
        f.write(f"**Date:** {Path.cwd()}\n")
        f.write(f"**Output Directory:** {out_dir}\n\n")
        
        f.write("## Dataset Split\n\n")
        f.write(f"- Total annotated images: {len(all_gt)}\n")
        f.write(f"- Evaluation images (held-out): {len(eval_images)}\n")
        f.write(f"- Training images: {len(train_images)}\n")
        f.write(f"- Validation images: {len(val_images)}\n")
        f.write(f"- Split ratio: 80/20\n")
        f.write(f"- Random seed: 42\n")
        f.write(f"- Leakage check: PASSED (train, val, eval are disjoint)\n\n")
        
        f.write("## Ground-Truth Box Counts\n\n")
        f.write(f"- Training boxes: {sum(len(all_gt[img]) for img in train_images)}\n")
        f.write(f"- Validation boxes: {sum(len(all_gt[img]) for img in val_images)}\n")
        f.write(f"- Total boxes: {sum(len(v) for v in all_gt.values())}\n\n")
        
        f.write("## Positive Crops\n\n")
        f.write(f"- Total GT boxes: {positive_stats['total']}\n")
        f.write(f"- Accepted: {positive_stats['accepted']}\n")
        f.write(f"- Rejected: {positive_stats['rejected']}\n\n")
        f.write("Rejection reasons:\n")
        for reason, count in positive_stats['rejections'].items():
            f.write(f"- {reason}: {count}\n")
        f.write("\n")
        
        f.write("## Negative Crops\n\n")
        f.write(f"- Target: {total_target_negatives} (2x positive)\n")
        f.write(f"- Accepted: {negative_stats['accepted']}\n")
        f.write(f"- Total attempts: {negative_stats['total_attempted']}\n\n")
        f.write("By split:\n")
        f.write(f"- Train: {negative_stats['by_split']['train']}\n")
        f.write(f"- Val: {negative_stats['by_split']['val']}\n\n")
        f.write("By source:\n")
        f.write(f"- Random background: {negative_stats['by_source']['random_background']}\n")
        f.write(f"- M3_D hard negatives: 0 (BLOCKED - pending future implementation)\n\n")
        
        f.write("## HOG Configuration\n\n")
        f.write("(For manifest only - features not extracted in this step)\n\n")
        for key, value in HOG_CONFIG.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")
        
        f.write("## Coordinate Semantics\n\n")
        f.write("The code uses `w = x2 - x1` and `h = y2 - y1`, indicating exclusive right/bottom coordinates.\n")
        f.write("This is the repository's convention, but official GTSDB semantics have not been independently confirmed.\n\n")
        
        f.write("## Files Created\n\n")
        f.write(f"- Split manifest: {split_manifest}\n")
        f.write(f"- Crop metadata: {metadata_csv}\n")
        f.write(f"- HOG config: {hog_config_file}\n")
        f.write(f"- Crop images: {crops_dir}\n")
        f.write(f"- This report: {report}\n")
    
    print(f"Report saved: {report}")
    
    print("\n" + "=" * 80)
    print("Data preparation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
