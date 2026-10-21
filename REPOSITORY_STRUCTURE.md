# Traffic Sign Detection Project - Repository Structure

This document describes the folder structure and how to use the pipeline.

## Folder Structure

```
Image_processing_project/
├── src/                          # Core pipeline implementation
│   ├── config.py                # Configuration parameters (C8 baseline)
│   ├── pipeline.py              # Pipeline orchestration
│   ├── stage1_enhancement/      # Image enhancement modules
│   │   ├── contrast.py
│   │   ├── denoising.py
│   │   └── noise_analysis.py
│   ├── stage2_segmentation/     # Color segmentation modules
│   │   ├── color_spaces.py
│   │   ├── morphology.py
│   │   └── thresholding.py
│   ├── visualizer.py
│   ├── run_phase_l_expc.py     # Candidate generation
│   └── run_phase_m2_experiment.py # Color rules
├── models/                      # Model training scripts
│   ├── train_hog_svm.py         # Train HOG+SVM model
│   ├── load_hog_svm.py          # Load and use model
│   └── prepare_training_crops.py # Prepare training data
├── evaluation/                  # Evaluation scripts
│   ├── run_c8_custom_image_list.py  # Main C8 evaluator
│   ├── run_c8_heldout_evaluation.py # Held-out evaluator
│   └── run_stage2_experiment_c8.py  # Original C8 validation
├── diagnostics/                 # Diagnostic scripts (not committed)
│   ├── app.py
│   └── evaluate_and_compare.py
├── scripts/                     # Historical experiments (not committed)
│   ├── experiments/
│   └── diagnostics/
├── results/                     # Detection outputs (not committed)
│   └── detection_output/
├── data/                        # Dataset (downloaded separately)
│   ├── split_manifest.csv       # Train/val split manifest
│   └── TrainIJCNN2013/          # GTSDB dataset (downloaded)
├── main.py                      # Main entry point
├── download_data.py             # Dataset download script
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore patterns
└── .gitattributes               # Git LFS patterns for large files
```

## Quick Start for Your Friend

### 1. Clone the repository
```bash
git clone https://github.com/Tejas-coder-cloud/Image_processing_project.git
cd Image_processing_project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
```bash
python download_data.py
```

This will download the GTSDB dataset to `data/TrainIJCNN2013/`.

### 4. Train the HOG+SVM model (optional)
```bash
python models/prepare_training_crops.py
python models/train_hog_svm.py
```

Note: The trained model files are in `experiments_phase_n1/hog_svm_v1/` and are referenced by the evaluation scripts.

### 5. Run the pipeline
```bash
# Run on validation set (98 images)
python main.py validation

# Run on combined set (118 images)
python main.py combined

# Run on held-out set (20 images)
python main.py heldout
```

## Output Location

Detection results and visualizations are saved to:
```
results/detection_output/
```

**Important:** The output directory is not committed to Git to keep the repository lightweight.

## Dataset Location

**Important:** The dataset images are not committed to Git. They must be downloaded separately using `download_data.py`.

Dataset location after download:
```
data/TrainIJCNN2013/
├── 00000.ppm
├── 00001.ppm
├── ...
├── 00599.ppm
└── gt.txt
```

## Model Files

The trained HOG+SVM model files are located in:
```
experiments_phase_n1/hog_svm_v1/
├── svm_model.joblib
├── scaler.joblib
├── hog_config.json
├── training_info.json
└── training_report.md
```

These files are referenced by the evaluation scripts but are not committed to Git due to their size.

## C8 Baseline Configuration

The C8 baseline configuration is in `src/config.py`:
- LAB red a* threshold: 130
- LAB blue b* threshold: 155
- LAB yellow b* threshold: 140
- HSV saturation: S=70 for red, blue, yellow
- Stage 1 sharpening: 0.35
- Canny thresholds: (30, 100)
- Color AR deviation: 0.25
- Edge AR deviation: 0.30
- NMS IoU threshold: 0.40
- SVM decision threshold: +0.2

## Pipeline Stages

1. **Stage 1: Image Enhancement**
   - Noise estimation and denoising
   - CLAHE contrast enhancement
   - Unsharp sharpening

2. **Stage 2: Color Segmentation**
   - HSV and LAB color space conversion
   - Multi-color segmentation (red, blue, yellow)
   - Morphological operations
   - Candidate region extraction

3. **Classification**
   - HOG feature extraction (64x64, 9 orientations)
   - SVM classification with RBF kernel
   - Decision threshold filtering

## Evaluation Metrics

C8 Baseline Performance (Validation Set - 98 images):
- Precision: 60.33%
- Recall: 44.51%
- F1 Score: 51.23%

C8 Baseline Performance (Combined Set - 118 images):
- Precision: 59.72%
- Recall: 44.56%
- F1 Score: 51.04%
