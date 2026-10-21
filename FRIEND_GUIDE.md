# Quick Start Guide for Your Friend

## What This Project Is

This is a **Traffic Sign Detection and Recognition** project using classical computer vision (no deep learning/CNNs). It detects and classifies traffic signs in road scene images using:
- Color segmentation (HSV + LAB color spaces)
- Geometric filtering (aspect ratio, symmetry, solidity)
- HOG feature extraction
- Two SVM classifiers

**Current Scope:** Detection + Verification + Recognition
- ✅ **Detection:** Finds WHERE traffic signs are (traditional IP)
- ✅ **Verification:** SVM 1 distinguishes "sign vs background"
- ✅ **Recognition:** SVM 2 identifies sign type (43 classes)
- ✅ **Full Pipeline:** Traditional IP → SVM 1 (binary) → SVM 2 (multi-class)

**Final Pipeline Results (98 validation images):**
- Detection: Precision 66.96%, Recall 45.73%, F1 54.35%
- Recognition: 76.00% accuracy on 43 sign classes

**Architecture:**
1. Traditional image processing for detection (color + geometry)
2. SVM 1: Binary verification (reduces false positives)
3. SVM 2: Multi-class recognition (identifies sign type)

---

## Prerequisites

1. **Python 3.8+** installed
2. **Git** installed
3. Clone the repository:
   ```bash
   git clone https://github.com/Tejas-coder-cloud/Image_processing_project.git
   cd Image_processing_project
   ```

---

## Setup Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- opencv-python
- numpy
- scikit-learn
- scikit-image
- pandas
- joblib

### 2. Download the Dataset
```bash
python download_data.py
```

This downloads the GTSDB dataset (~400MB) to:
```
data/TrainIJCNN2013/
├── 00000.ppm through 00599.ppm (600 images)
└── gt.txt (ground truth annotations)
```

**Note:** If the download script doesn't work, download manually from:
https://sid.erda.net/public/sid/D32E2FCB-7CC9-4903-9B32-2E46969A9DD8/TrainIJCNN2013.zip
Extract to `data/TrainIJCNN2013/`

### 3. Get the Trained Models

**Important:** The repository does NOT include the trained SVM models. You need to obtain them or train your own.

**Two models are required:**

1. **SVM 1: Binary Verification Model** (for filtering false positives)
   - Location: `experiments_phase_n1/hog_svm_v1/`
   - Purpose: Reduce false positives in detection
   - Used by: Two SVMs pipeline (main.py validation)
   - NOT used by: IP-only pipeline (main.py ip_only)

2. **SVM 2: Recognition Model** (for identifying sign types)
   - Location: `models/recognition_svm_v1/`
   - Purpose: Identify which of 43 sign types each detection is
   - Used by: Both pipelines
   - Can be trained (see step 4 below)

**Option A: Get models from the project owner**
- Ask for `experiments_phase_n1/hog_svm_v1/` folder (SVM 1)
- Ask for `models/recognition_svm_v1/` folder (SVM 2) OR train it yourself
- Extract to the project root

**Option B: Train your own models**
```bash
# Train SVM 1 (binary verification) - requires time
python models/prepare_training_crops.py
python models/train_hog_svm.py

# Train SVM 2 (recognition) - always required
python models/train_recognition_svm.py
```

### 4. Train Recognition SVM (Required for Recognition)
**Train the multi-class SVM for sign recognition:**
```bash
python models/train_recognition_svm.py
```

This trains SVM 2 on GTSRB dataset (43 sign classes, ~39,000 images).
Training takes several minutes and saves the model to `models/recognition_svm_v1/`.

**Note:** The GTSRB dataset is already in `data/Train/` (43 class folders with cropped sign images).

---

## How to Run the Pipeline

### Pipeline Options

**Option 1: Two SVMs (Recommended)**
- Traditional IP detection
- SVM 1: Binary verification (sign vs background)
- SVM 2: Multi-class recognition (43 sign types)
- Better performance: F1=54.35%, Recognition=76%

```bash
# Run on validation set (98 images)
python main.py validation

# Run on custom images
python main.py custom --images 00000.ppm,00001.ppm
```

**Option 2: IP-Only + SVM Recognition (Instructor Requirements)**
- Traditional IP detection only (NO SVM verification)
- SVM: Multi-class recognition (43 sign types)
- Follows instructor requirements exactly
- Lower performance: F1=5.28%, Recognition=72.5%

```bash
# Run on validation set (98 images)
python main.py ip_only
```

### What Each Pipeline Does

**Two SVMs (main.py validation):**
- Traditional IP detection (color + geometry)
- SVM 1 verification (filters false positives)
- SVM 2 recognition (identifies sign type)
- Output: High precision, reasonable recall

**IP-Only (main.py ip_only):**
- Traditional IP detection (color + geometry)
- NO SVM verification (all candidates kept)
- SVM recognition (identifies sign type)
- Output: Low precision, higher recall (many false positives)

---

## What Outputs to Expect

After running the pipeline, check these directories:

**Two SVMs (main.py validation):**
```
results/detection_output/final_pipeline/
├── metrics.json              # Detection + recognition metrics
└── visual_overlays/         # PNG images with detection overlays
    ├── 00024.png
    ├── 00042.png
    └── ... (98 images)
```

**IP-Only (main.py ip_only):**
```
results/detection_output/ip_only_svm_recognition/
├── metrics.json              # Detection + recognition metrics
└── visual_overlays/         # PNG images with detection overlays
    ├── 00024.png
    ├── 00042.png
    └── ... (98 images)
```
    ├── ... (one per image)
    └── 00554.png
```

### Visual Overlays
Each PNG overlay shows:
- **Green boxes** = Ground truth traffic signs (from dataset)
- **Blue boxes** = Correctly detected signs (True Positives)
- **Red boxes** = Incorrect detections (False Positives)

### Metrics (metrics.json)
```json
{
  "images_evaluated": 98,
  "gt_boxes": 164,
  "total_detections": 112,
  "tp": 75,
  "fp": 37,
  "fn": 89,
  "precision": 0.6696,
  "recall": 0.4573,
  "f1": 0.5435,
  "svm_used": true
}
```

**Note:** These are **detection metrics** (how well we find signs), NOT recognition metrics (how well we identify sign types). Recognition is not yet implemented.

---

## What to Check/Verify

### 1. Dataset is Downloaded
Check that images exist:
```bash
ls data/TrainIJCNN2013/TrainIJCNN2013/
```
You should see 600 PPM files (00000.ppm through 00599.ppm)

### 2. Model Files Exist
Check that the SVM model is available:
```bash
ls experiments_phase_n1/hog_svm_v1/
```
You should see:
- svm_model.joblib
- scaler.joblib
- hog_config.json
- training_info.json
- training_report.md

**If these are missing, the pipeline will skip SVM filtering and produce poor results.**

### 3. Pipeline Runs Successfully
Run the validation set and check for errors:
```bash
python main.py validation
```

You should see output like:
```
Using validation images: 98
Total GT boxes: 164
SVM model loaded successfully

Processing images with C8 pipeline...
  00024.ppm: 0 detections, TP=0, FP=0, FN=1
  ...

Overall Metrics:
  TP: 75, FP: 37, FN: 89
  Precision: 66.96%
  Recall: 45.73%
  F1: 54.35%
```

### 4. Visual Overlays Generated
Check that overlays were created:
```bash
ls results/detection_output/c8_full_eval/visual_overlays/
```

Open a few PNG files to verify:
- Green boxes show where traffic signs actually are
- Blue boxes show correctly detected signs
- Red boxes show false positives (should be minimal)

---

## Expected Performance

With the trained model, you should see approximately:
- **Precision:** 60-67% (60-67% of detections are correct)
- **Recall:** 44-46% (44-46% of actual signs are detected)
- **F1:** 51-54%

**Without the SVM model**, performance will be much worse (many false positives).

---

## What Work Is Left (Future Improvements)

The current implementation includes **detection + verification + recognition**. The following features could be improved:

### 1. Recognition Accuracy (Currently 76%)
**Status:** ✅ Implemented, but can be improved

**Current performance:**
- Recognition accuracy: 76% on validation set
- Trained on GTSRB with 99.66% validation accuracy
- Lower accuracy on detection due to imperfect bounding boxes

**Potential improvements:**
- Better bounding box alignment before recognition
- Data augmentation for training
- Ensemble methods
- Fine-tuning on detected crops

### 2. Detection Performance
**Status:** ✅ Implemented, can be tuned

**Current performance:**
- Detection F1: 54.35%
- Recall: 45.73% (many signs missed)

**Potential improvements:**
- Tune LAB/HSV thresholds for different lighting
- Add more sophisticated geometric filters
- Improve edge symmetry computation
- Add temporal filtering for video sequences

### 3. Performance Improvements
**Potential areas:**
- Tune LAB/HSV thresholds for different lighting conditions
- Add more sophisticated geometric filters
- Improve edge symmetry computation
- Add temporal filtering for video sequences

### 3. Dataset Expansion
**Current dataset:** GTSDB (600 full-frame images, 851 annotated signs)
**Potential additions:**
- GTSRB for recognition training
- Custom datasets for specific use cases
- Synthetic data augmentation

### 4. Evaluation Metrics
**Current metrics:** Precision, Recall, F1 (binary detection)
**Potential additions:**
- Per-class accuracy (for recognition)
- mAP (mean Average Precision)
- Confusion matrix
- ROC curves

---

## Troubleshooting

### "SVM model not found" Error
**Problem:** The model files are missing.
**Solution:** Get the `experiments_phase_n1/hog_svm_v1/` folder from the project owner.

### "Failed to load image" Error
**Problem:** Dataset not downloaded correctly.
**Solution:** Run `python download_data.py` again or download manually.

### "Module not found" Error
**Problem:** Dependencies not installed.
**Solution:** Run `pip install -r requirements.txt`

### Poor Performance (many false positives)
**Problem:** SVM model not loaded or missing.
**Solution:** Verify model files exist in `experiments_phase_n1/hog_svm_v1/`

---

## Important Notes

1. **Images are NOT in Git** - The dataset is too large. You must download it separately.
2. **Model is NOT in Git** - The trained SVM model must be obtained separately.
3. **Results are NOT in Git** - Outputs are generated when you run the pipeline.
4. **This is classical CV** - No CNNs, YOLO, or deep learning. Pure traditional computer vision.
5. **C8 is the baseline** - Current configuration uses LAB threshold relaxation for better recall.

---

## Project Structure

```
Image_processing_project/
├── src/                          # Core pipeline code
│   ├── config.py                # C8 configuration
│   ├── pipeline.py              # Pipeline orchestration
│   ├── stage1_enhancement/      # Image enhancement
│   └── stage2_segmentation/     # Color segmentation
├── models/                      # Model training scripts
├── evaluation/                  # Evaluation scripts
│   └── c8_full_eval.py         # Main evaluator (use this)
├── data/                        # Dataset (download separately)
│   ├── split_manifest.csv      # Train/val split
│   └── TrainIJCNN2013/         # GTSDB images
├── main.py                      # Entry point
├── download_data.py             # Dataset downloader
├── requirements.txt             # Dependencies
└── README.md                    # Full documentation
```

---

## Contact

If you have issues:
1. Check the dataset is downloaded correctly
2. Verify the SVM model files exist
3. Check the visual overlays to understand what's being detected
4. Review the metrics.json file for performance numbers

Good luck with the traffic sign detection! 🚦
