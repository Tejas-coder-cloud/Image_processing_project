# Quick Start Guide for Your Friend

## What This Project Is

This is a **Traffic Sign Detection** project using classical computer vision (no deep learning/CNNs). It detects traffic signs in road scene images using:
- Color segmentation (HSV + LAB color spaces)
- Geometric filtering (aspect ratio, symmetry, solidity)
- HOG feature extraction
- SVM classification

**Current Baseline:** C8 configuration with LAB threshold relaxation
- **Validation Performance:** Precision 66.96%, Recall 45.73%, F1 54.35%

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

### 3. Get the Trained Model

**Important:** The repository does NOT include the trained SVM model. You need to obtain it from the project owner or train your own.

**Option A: Get the model from the project owner**
- Ask for the `experiments_phase_n1/hog_svm_v1/` folder
- Extract it to the project root

**Option B: Train your own model**
```bash
# This requires time and will produce different results
python models/prepare_training_crops.py
python models/train_hog_svm.py
```

---

## How to Run the Pipeline

### Run on Validation Set (98 images)
```bash
python main.py validation
```

This will:
- Process 98 validation images
- Apply the full C8 pipeline (color segmentation → M3_D filtering → SVM)
- Generate visual overlays
- Save metrics

### Run on Custom Images
```bash
python main.py custom --images 00000.ppm,00001.ppm,00002.ppm
```

---

## What Outputs to Expect

After running the pipeline, check this directory:

```
results/detection_output/c8_full_eval/
├── metrics.json              # Overall performance metrics
└── visual_overlays/         # PNG images with detection overlays
    ├── 00024.png
    ├── 00042.png
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
