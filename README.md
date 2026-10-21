# Traffic Sign Detection - C8 Baseline

A classical computer vision pipeline for traffic sign detection using HOG features and SVM classification. The C8 baseline uses LAB threshold relaxation for improved recall.

## Quick Start

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
python scripts/download_data.py
```

This will download the GTSDB dataset to `data/TrainIJCNN2013/`.

### 4. Run the pipeline
```bash
# Run on validation set (98 images)
python main.py validation

# Run on combined set (118 images)
python main.py combined

# Run on held-out set (20 images)
python main.py heldout
```

## Results Location

Detection results and visualizations are saved to:
```
results/detection_output/
```

This directory contains:
- Detection metrics (TP, FP, FN, precision, recall, F1)
- Per-image detection results
- Visual overlays showing detected traffic signs

## Dataset

The project uses the German Traffic Sign Detection Benchmark (GTSDB):
- Training images: 600 PPM images (00000.ppm through 00599.ppm)
- Ground truth annotations: 851 labeled traffic signs
- Official split: 388 train, 98 validation, 20 held-out

## C8 Baseline Configuration

The C8 baseline uses the following key parameters:
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

## Project Structure

```
Image_processing_project/
├── src/                    # Core pipeline implementation
│   ├── config.py          # Configuration parameters
│   ├── pipeline.py        # Pipeline orchestration
│   ├── stage1_enhancement/ # Image enhancement modules
│   └── stage2_segmentation/ # Color segmentation modules
├── scripts/               # Utility and evaluation scripts
│   ├── download_data.py   # Dataset download script
│   ├── experiments/       # Historical experiment scripts
│   └── diagnostics/       # Diagnostic and analysis scripts
├── results/               # Detection outputs (generated)
├── data/                  # Dataset (downloaded separately)
├── main.py               # Main entry point
└── requirements.txt      # Python dependencies
```

## Dependencies

- OpenCV (cv2)
- NumPy
- scikit-learn
- scikit-image
- pandas
- joblib

## License

This project uses the GTSDB dataset. Please refer to the dataset license for usage terms.
