"""Quick test on multiple problematic images."""
import cv2
from src.config import PipelineConfig
from src.pipeline import run_full_pipeline
import os

test_images = [
    ("102 (mandatory blue)", "gtsdb_dataset/train/images/102_jpg.rf.00c2df28c369ba2617fb4d860e3d8e1c.jpg"),
    ("101 (road signs)",     "gtsdb_dataset/train/images/101_jpg.rf.195906d75cfd98436709ee9fd1ef4a2b.jpg"),
    ("100 (speed overpass)", "gtsdb_dataset/train/images/100_jpg.rf.147e4e01d9606f30e6521fd78d0d05ec.jpg"),
    ("122 (speed limit 50)", "gtsdb_dataset/train/images/122_jpg.rf.147e4e01d9606f30e6521fd78d0d05ec.jpg") if os.path.exists("gtsdb_dataset/train/images/122_jpg.rf.147e4e01d9606f30e6521fd78d0d05ec.jpg") else None,
]

import glob
# Find a 122_jpg image
matches_122 = glob.glob("gtsdb_dataset/train/images/122_jpg*")
if matches_122:
    test_images[3] = ("122 (speed limit 50)", matches_122[0])

config = PipelineConfig()

os.makedirs("outputs/debug", exist_ok=True)
for item in test_images:
    if item is None:
        continue
    name, path = item
    img = cv2.imread(path)
    if img is None:
        print(f"  SKIP: {name}")
        continue
    res = run_full_pipeline(img, config)
    bboxes = res["stage2"]["bounding_boxes"]
    print(f"  {name}: {len(bboxes)} detections")
    for i, b in enumerate(bboxes):
        print(f"    Box #{i+1}: x={b[0]}, y={b[1]}, w={b[2]}, h={b[3]}")
    
    safe_name = name.split("(")[0].strip().replace(" ", "_")
    cv2.imwrite(f"outputs/debug/test_{safe_name}_overlay.png", res["stage2"]["overlay"])
    cv2.imwrite(f"outputs/debug/test_{safe_name}_mask.png", res["stage2"]["binary_mask_cleaned"])
    print(f"    Saved to outputs/debug/test_{safe_name}_overlay.png")
