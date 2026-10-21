# Final Pipeline Summary

## Architecture Overview

This project implements a traffic sign detection and recognition system using classical computer vision and SVM classifiers, following the instructor's requirement: **Traditional IP for detection, SVM for recognition**.

### Pipeline Stages

```
1. Traditional Image Processing Detection
   ├─ Stage 1: Image Enhancement
   │  ├─ Noise estimation and denoising
   │  ├─ CLAHE contrast enhancement
   │  └─ Unsharp sharpening
   ├─ Stage 2: Color Segmentation
   │  ├─ HSV + LAB color space conversion
   │  ├─ Multi-color segmentation (red, blue, yellow)
   │  └─ Morphological operations
   ├─ M2_D Geometric Filtering
   │  ├─ Solidity >= 0.20
   │  ├─ Edge symmetry >= 0.12
   │  └─ AR deviation <= 0.25
   └─ M3_D Geometric Filtering
      ├─ Edge symmetry >= 0.18
      ├─ AR deviation <= 0.30
      └─ Area <= 15000
   ↓ (~300-500 candidates)

2. SVM 1: Binary Verification
   ├─ Purpose: Filter false positives (sign vs background)
   ├─ Input: HOG features from candidate crops
   ├─ Model: Binary SVM (RBF kernel)
   ├─ Threshold: >= 0.2
   └─ Output: ~100-120 verified candidates
   ↓

3. SVM 2: Multi-Class Recognition
   ├─ Purpose: Identify sign type (43 classes)
   ├─ Input: HOG features from verified crops
   ├─ Model: Multi-class SVM (RBF kernel, one-vs-rest)
   ├─ Classes: 43 sign types (stop, speed limits, etc.)
   └─ Output: Class ID for each detection
   ↓

4. Final Output
   ├─ Bounding boxes with class IDs
   ├─ Detection metrics (TP, FP, FN, precision, recall, F1)
   └─ Recognition accuracy
```

---

## Results (98 Validation Images)

### Detection Metrics
- **TP:** 75 (correctly detected signs)
- **FP:** 37 (false positive detections)
- **FN:** 89 (missed signs)
- **Precision:** 66.96% (66.96% of detections are correct)
- **Recall:** 45.73% (45.73% of actual signs detected)
- **F1 Score:** 54.35%

### Recognition Metrics
- **Correct recognitions:** 57/75
- **Recognition accuracy:** 76.00%
- **Classes:** 43 sign types

### Comparison: Single SVM vs Two SVMs

| Metric | Single SVM (Binary Only) | Two SVMs (Binary + Recognition) |
|--------|-------------------------|--------------------------------|
| Detection F1 | 54.35% | 54.35% (unchanged) |
| Recognition | N/A | 76.00% accuracy |
| Architecture | Incomplete (no recognition) | Complete (detection + recognition) |

---

## Model Details

### SVM 1: Binary Verification
- **Location:** `experiments_phase_n1/hog_svm_v1/`
- **Type:** Binary SVM (sign vs background)
- **Kernel:** RBF
- **Features:** HOG (64x64, 9 orientations)
- **Purpose:** Reduce false positives
- **Decision threshold:** 0.2

### SVM 2: Multi-Class Recognition
- **Location:** `models/recognition_svm_v1/`
- **Type:** Multi-class SVM (43 classes)
- **Kernel:** RBF
- **Features:** HOG (64x64, 9 orientations)
- **Training data:** GTSRB (39,209 images, 43 classes)
- **Validation accuracy:** 99.66%
- **Purpose:** Identify sign type
- **Decision function:** One-vs-rest

---

## Instructor Requirements Compliance

### Requirement: Traditional IP for Detection
✅ **COMPLIANT**
- Detection uses only traditional image processing
- Color segmentation (HSV + LAB)
- Geometric filtering (aspect ratio, symmetry, solidity)
- No deep learning or CNNs used for detection

### Requirement: SVM for Recognition
✅ **COMPLIANT**
- SVM 2 performs multi-class recognition
- 43 sign classes identified
- Recognition accuracy: 76%
- Trained on GTSRB dataset

---

## Files and Usage

### Training Recognition SVM
```bash
python models/train_recognition_svm.py
```
- Trains on GTSRB dataset in `data/Train/`
- Saves model to `models/recognition_svm_v1/`
- Takes several minutes (39,209 images)

### Running Final Pipeline
```bash
# Validation set (98 images)
python main.py validation

# Custom images
python main.py custom --images 00000.ppm,00001.ppm
```

### Output Location
```
results/detection_output/final_pipeline/
├── metrics.json              # Detection + recognition metrics
└── visual_overlays/         # PNG overlays with class IDs
    ├── 00024.png
    ├── 00042.png
    └── ... (98 images)
```

---

## Visual Overlays

Each overlay shows:
- **Green boxes:** Ground truth with class ID (e.g., "GT:12")
- **Blue boxes:** True positive with predicted class (e.g., "TP cls:12")
- **Red boxes:** False positive with predicted class (e.g., "FP cls:5")

---

## Performance Analysis

### Detection Performance (F1: 54.35%)
**Strengths:**
- High precision (66.96%) - few false positives
- Maintained by SVM 1 verification

**Weaknesses:**
- Low recall (45.73%) - many signs missed
- Could be improved with threshold tuning

### Recognition Performance (Accuracy: 76%)
**Strengths:**
- Good accuracy given imperfect bounding boxes
- SVM 2 trained with 99.66% on GTSRB

**Weaknesses:**
- 24% error rate on detected signs
- Likely due to:
  - Imperfect bounding box alignment
  - Different image characteristics between GTSRB (cropped) and GTSDB (full-frame)
  - Sign occlusion or partial visibility

---

## Key Achievements

1. ✅ **Complete pipeline** - Detection + Verification + Recognition
2. ✅ **Instructor compliance** - Traditional IP for detection, SVM for recognition
3. ✅ **Two-SVM architecture** - Maintains performance while adding recognition
4. ✅ **43-class recognition** - Identifies all GTSDB sign types
5. ✅ **Visual outputs** - Clear overlays showing GT and predictions with class IDs
6. ✅ **Reproducible** - Fixed configuration, documented pipeline

---

## Dataset Requirements

### Required Datasets
1. **GTSDB** (Detection)
   - Full-frame road scenes
   - Location: `data/TrainIJCNN2013/`
   - 600 images, 851 annotated signs

2. **GTSRB** (Recognition)
   - Cropped sign images
   - Location: `data/Train/`
   - 43 classes, ~39,000 images
   - Already present in project

### Required Models
1. **SVM 1** (Verification)
   - Location: `experiments_phase_n1/hog_svm_v1/`
   - Must be obtained from project owner

2. **SVM 2** (Recognition)
   - Location: `models/recognition_svm_v1/`
   - Trained using `models/train_recognition_svm.py`

---

## Conclusion

The final pipeline successfully implements:
- **Traditional IP detection** (as required by instructor)
- **SVM-based recognition** (as required by instructor)
- **Two-SVM architecture** for optimal performance
- **Complete end-to-end system** with visual outputs

**Final F1 Score:** 54.35% (detection)
**Recognition Accuracy:** 76.00% (43 classes)

This architecture balances the instructor's requirements with practical performance considerations.
