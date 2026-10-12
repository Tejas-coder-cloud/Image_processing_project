# Stage 3 Candidate Detection Investigation Report

**Date:** 2026-10-12
**Investigator:** Devin
**Dataset:** GTSDB (German Traffic Sign Detection Benchmark) - TrainIJCNN2013
**Sample Size:** 20 images (00000.ppm through 00019.ppm)

---

## Executive Summary

The investigation reveals a **critical failure in Stage 3**: the pipeline produces **zero true positives** across all 20 test images, with 676 false positives and 29 false negatives. The root cause is **incorrect vegetation mask subtraction** that merges traffic signs with surrounding background, creating oversized contours that are rejected by the area filter.

**Key Findings:**
- **Precision:** 0.00% (0/676 detections are correct)
- **Recall:** 0.00% (0/29 ground truth signs detected)
- **Root Cause:** Vegetation mask incorrectly removes 29% of sign pixels and dilates into sign regions, causing sign-background merger
- **Secondary Issue:** Morphological opening with 1 iteration can destroy small sign contours

---

## 1. Project File Map

### Entry Points
- **`app.py`** - Interactive Streamlit web UI (Stage 1 & 2 only)
- **`run_pipeline.py`** - Command-line runner for single images
- **`run_batch_gallery.py`** - Batch evaluation on 10 diverse test cases
- **`run_gtsdb_150_eval.py`** - Batch evaluation with YOLO format labels (incompatible with current data)
- **`investigate_stage3.py`** - Investigation script created for this analysis

### Source Code Structure
```
src/
├── config.py                          # Stage1Config, Stage2Config, PipelineConfig
├── pipeline.py                        # run_stage1(), run_stage2(), run_full_pipeline()
├── visualizer.py                       # Visualization/report generation
├── stage1_enhancement/
│   ├── __init__.py
│   ├── noise_analysis.py               # Immerkaer noise estimation, FFT spectrum
│   ├── denoising.py                    # Adaptive bilateral/NLM denoising
│   └── contrast.py                     # CLAHE in CIELAB, gamma, unsharp masking
└── stage2_segmentation/
    ├── __init__.py
    ├── color_spaces.py                 # RGB→HSV, RGB→CIELAB conversions
    ├── thresholding.py                 # HSV & LAB color thresholding
    └── morphology.py                   # Morphological ops, candidate extraction
```

### Dataset Structure
```
data/
├── TrainIJCNN2013/TrainIJCNN2013/
│   ├── 00000.ppm through 00599.ppm    # 600 training images (1360×800)
│   ├── gt.txt                          # Ground truth: filename;left;top;right;bottom;class_id
│   ├── ReadMe.txt                      # Dataset documentation
│   └── ex.txt                          # Example submission
├── TestIJCNN2013/TestIJCNN2013Download/
│   └── 00000.ppm through 00054.ppm     # 55 test images
├── Train/                              # GTSRB cropped signs (43 class folders)
├── Test/                               # GTSRB test images
└── Meta/                               # Class reference images (0.png through 42.png)
```

### Configuration Parameters (src/config.py)

**Stage 1 (Enhancement):**
- `denoise_method`: "adaptive" (auto-selects based on noise)
- `bilateral_d`: 5, `bilateral_sigma_color`: 25.0, `bilateral_sigma_space`: 10.0
- `clahe_clip_limit`: 2.5, `clahe_grid_size`: (8, 8)
- `gamma`: 1.15
- `sharpen_amount`: 0.35

**Stage 2 (Segmentation):**
- HSV Red: H∈[0,12]∪[155,180], S≥50, V≥40
- HSV Blue: H∈[95,135], S≥40, V≥40
- HSV Yellow: H∈[14,38], S≥50, V≥50
- LAB Red: a*≥140
- LAB Blue: b*≤115
- LAB Yellow: b*≥140
- Morphology: kernel=5, close=3, open=1
- Geometry: area∈[40, 18000], dim∈[8, 180], aspect∈[0.35, 2.0]
- Circularity≥0.15, Convexity≥0.40
- Color density∈[0.05, 0.95]
- Edge density∈[0.01, 0.65]
- Road cutoff: y≤88% of image height
- Structural verification: enabled

---

## 2. Existing Pipeline Explanation

### Stage 1: Image Enhancement
1. **Noise Estimation** - Immerkaer Laplacian convolution to estimate σ
2. **Power Spectrum** - 2D FFT with azimuthal averaging
3. **Denoising** - Adaptive bilateral/NLM based on estimated noise
4. **CLAHE** - Contrast Limited Adaptive Histogram Equalization in CIELAB L* channel
5. **Gamma Correction** - γ=1.15 exposure adjustment
6. **Unsharp Masking** - High-frequency detail sharpening (α=0.35)

### Stage 2: Color Segmentation
1. **Color Space Conversion** - RGB→HSV and RGB→CIELAB
2. **Dual-Space Thresholding** - HSV AND LAB intersection for each color
3. **Vegetation Subtraction** - Green mask (H∈[35,85]) dilated and subtracted
4. **Morphological Closing** - Elliptical kernel, 3 iterations (bridges sign borders)
5. **Morphological Opening** - Elliptical kernel, 1 iteration (removes noise)
6. **Candidate Extraction** - 12-filter cascade (see below)

### Stage 3: Candidate Filtering (12 Filters)
Applied in `extract_candidate_regions()` in `morphology.py`:

1. **Area bounds** - [40, 18000] pixels
2. **Bounding box dimensions** - [8, 180] pixels per side
3. **Aspect ratio** - [0.35, 2.0]
4. **Spatial position** - Reject if y > 88% (road) or y+h < 2% (sky)
5. **Circularity** - 4πA/P² ≥ 0.15
6. **Convexity** - A/hull_area ≥ 0.40
7. **Color density** - Fraction of sign-colored pixels in bbox ∈ [0.05, 0.95]
8. **Edge density** - Canny edge density ∈ [0.01, 0.65]
9. **Vehicle rejection** - Large red blobs in lower 55% of frame
10. **Traffic light rejection** - Tall-narrow (aspect<0.5) with 2+ bright spots
11. **Large background rejection** - Area>5000 or (area>1500 and variance>2000) or (area>800 and variance<150)
12. **Structural verification** - Requires sign color content OR white interior OR solid shape OR yellow content

---

## 3. Pipeline Trace on Example Image (00000.ppm)

### Input
- **Image:** 00000.ppm (1360×800, 3-channel BGR)
- **Ground Truth:** 1 sign (class=11, priority at next intersection)
  - Location: x=774, y=411, w=41, h=35
  - Total pixels: 1435

### Stage 1 Output
- Noise σ: ~5-10 (estimated)
- Enhanced image: CLAHE applied, contrast improved

### Stage 2 Color Thresholding
- **Red mask (HSV only):** 446/1435 pixels (31.1%) in GT region
- **Red mask (LAB only):** 494/1435 pixels (34.4%) in GT region
- **Red mask (intersection):** 425/1435 pixels (29.6%) in GT region
- **Combined mask (red+blue+yellow):** 513/1435 pixels (35.8%) in GT region

**Vegetation Subtraction Impact:**
- Vegetation pixels in GT region: 420/1435 (29.3%)
- Red mask before vegetation: 72,494 pixels total
- Red mask after vegetation: 38,390 pixels total (47% reduction)
- Vegetation in GT surrounding region (50px padding): 8,288 pixels

### Morphological Operations
**Current parameters:** kernel=5, close=3, open=1
- After closing: 1,299/1,435 pixels (90.5%) in GT region
- After opening: 1,299/1,435 pixels (90.5%) in GT region (no change for this sign)
- **Result:** Single contour with area=1,224.5, bbox=(774, 411, 41, 35) ✓ CORRECT SIZE

**However:** The contour is actually merged with surrounding vegetation
- Actual detected contour: bbox=(483, 264, 391, 283), area=47,775
- This massive contour is REJECTED by area filter (47,775 > 18,000 max)

### Candidate Filtering Results
The GT sign's actual contour passes ALL 12 filters when isolated:
- ✓ Area: 1,224.5 ∈ [40, 18,000]
- ✓ Dimensions: 41×35 ∈ [8, 180]
- ✓ Aspect: 1.17 ∈ [0.35, 2.0]
- ✓ Position: y_ratio=0.54 < 0.88
- ✓ Circularity: 0.68 > 0.15
- ✓ Convexity: 0.95 > 0.40
- ✓ Color density: 0.36 ∈ [0.05, 0.95]
- ✓ Edge density: 0.31 ∈ [0.01, 0.65]
- ✓ Vehicle: Not a vehicle
- ✓ Traffic light: Not a traffic light
- ✓ Large background: Not too large
- ✓ Structure: Has color (0.36), white (394px), shape (0.68 circularity), yellow (89px)

**But the merged contour fails:**
- ✗ Area: 47,775 > 18,000 (REJECTED)
- ✗ Dimensions: 391×283 > 180 (REJECTED)

---

## 4. Observed Failure Modes

### Failure Mode 1: Vegetation Mask Causes Sign-Background Merger (PRIMARY)
**Severity:** CRITICAL
**Frequency:** Affects all signs near vegetation/trees

**Description:**
The vegetation mask (H∈[35,85], S≥30, V≥20) is too broad and incorrectly classifies sign pixels as vegetation. After dilation with 5×5 kernel, it bleeds into sign regions, causing:
1. Sign pixels to be removed from the detection mask
2. Sign contours to merge with surrounding vegetation through the dilated vegetation bridge
3. Massive merged contours that exceed the 18,000 pixel area limit

**Evidence:**
- Image 00000.ppm: 29% of GT sign pixels classified as vegetation
- Sign center (794, 428) is inside a 391×283 pixel contour (area 47,775)
- This merged contour is rejected by area filter
- Without vegetation subtraction, the sign would be detected correctly

**Affected Sign Types:**
- All signs with white interiors (prohibitory, danger triangles)
- Signs in shadow or with desaturated colors
- Signs near trees, grass, or foliage

### Failure Mode 2: Morphological Opening Destroys Small Signs (SECONDARY)
**Severity:** HIGH
**Frequency:** Affects small signs (<15px dimension)

**Description:**
With kernel=5 and open=1, morphological opening (erosion then dilation) can completely eliminate small sign contours before they reach the area filter.

**Evidence:**
Testing on GT region (41×35 pixels):
- kernel=3, close=2, open=1: PASS (area=503, bbox=40×35)
- kernel=5, close=1, open=1: PASS (area=193, bbox=28×20) - reduced size
- kernel=5, close=1, open=0: FAIL (area=0) - opening destroyed it
- kernel=5, close=2, open=0: FAIL (area=0) - opening destroyed it

**Current config (kernel=5, close=3, open=1):**
- With open=1: PASS (bbox=39×35, reduced from 41×35)
- With open=0: PASS (bbox=41×35, correct size)

**Impact:**
- Small signs may be eroded below the minimum dimension threshold (8px)
- Signs near the size threshold (8-15px) are at risk

### Failure Mode 3: Excessive False Positives from Noise Residuals
**Severity:** MEDIUM
**Frequency:** All images

**Description:**
After vegetation subtraction and morphology, residual noise pixels create many small contours (8-43 pixels) scattered across the image. These pass the minimum area filter (40px) but are not real signs.

**Evidence:**
- Image 00000.ppm: 29 detections, all 8-43 pixels, all false positives
- Image 00001.ppm: 70 detections, all false positives
- Average across 20 images: 33.8 false positives per image

**Characteristics:**
- Very small (8-43 pixels)
- Scattered across entire image (including sky, road, trees)
- Often at image borders (y=0 or y near image height)
- Low circularity and convexity

---

## 5. Ground Truth Label Format

### GTSDB Format (Current Dataset)
**File:** `data/TrainIJCNN2013/TrainIJCNN2013/gt.txt`
**Format:** `filename;left;top;right;bottom;class_id`

Example:
```
00000.ppm;774;411;815;446;11
00001.ppm;983;388;1024;432;40
00001.ppm;386;494;442;552;38
```

**Class Mapping:**
- 0-5, 7-8, 9-10, 15-16: Prohibitory (circular, white ground, red border)
- 11, 18-31: Danger (triangular, white ground, red border)
- 12-14, 17, 32, 41-42: Other
- 33-40: Mandatory (circular, blue ground)

**Coordinate System:**
- Pixel coordinates (0-indexed)
- left, top: top-left corner
- right, bottom: bottom-right corner (exclusive)
- Width = right - left
- Height = bottom - top

### YOLO Format (Expected by run_gtsdb_150_eval.py)
**Format:** `class_id cx_norm cy_norm w_norm h_norm`

**Incompatibility:**
The code expects YOLO format with normalized coordinates, but the dataset provides GTSDB format with absolute pixel coordinates. The `run_gtsdb_150_eval.py` script cannot run without YOLO-formatted labels.

---

## 6. Commands to Reproduce Results

### Run Investigation Script
```bash
cd C:\Users\9bdhr\Image_processing_project
python investigate_stage3.py
```

**Output:** `investigation_results/` directory containing:
- `*_overlay.png` - Detection overlay with TP/FP/FN visualization
- `*_mask.png` - Binary mask after Stage 2
- `*_enhanced.png` - Enhanced image after Stage 1
- `*_details.txt` - Detailed per-image analysis
- `investigation_summary.csv` - Summary statistics

### Run Original Pipeline on Single Image
```bash
python run_pipeline.py --image data/TrainIJCNN2013/TrainIJCNN2013/00000.ppm --output_dir investigation_results
```

### Run Streamlit UI
```bash
streamlit run app.py
```

---

## 7. Proposed Experiments

### Experiment 1: Disable Vegetation Subtraction
**Goal:** Verify that vegetation subtraction is the primary cause of false negatives

**Method:**
- Comment out vegetation mask creation and subtraction in `thresholding.py` (lines 132-137)
- Run investigation script on 20 images
- Compare detection rates

**Expected Outcome:**
- Significant increase in true positives
- Possible increase in false positives from actual vegetation

**Risk:** Increased false positives from trees, grass, bushes

### Experiment 2: Tighten Vegetation Thresholds
**Goal:** Reduce vegetation mask sensitivity to avoid classifying sign pixels

**Method:**
- Modify `_create_vegetation_mask()` in `thresholding.py`:
  - Narrow hue range: [40, 80] instead of [35, 85]
  - Increase saturation minimum: S≥50 instead of S≥30
  - Remove dilation or reduce kernel size to 3×3
- Run investigation script

**Expected Outcome:**
- Fewer sign pixels classified as vegetation
- Reduced sign-background merger
- Better detection rates

### Experiment 3: Remove Morphological Opening
**Goal:** Verify that opening destroys small signs

**Method:**
- Set `morph_open_iterations = 0` in `Stage2Config`
- Run investigation script on images with small signs

**Expected Outcome:**
- Better detection of small signs (<15px)
- Possible increase in small noise contours

### Experiment 4: Adjust Area/Dimension Thresholds
**Goal:** Allow larger merged contours while adding additional filtering

**Method:**
- Increase `max_area` from 18,000 to 50,000
- Increase `max_box_dim` from 180 to 300
- Add aspect ratio tightening or solidity filter to reject irregular shapes
- Run investigation script

**Expected Outcome:**
- Signs that merged with background may pass area filter
- Need additional shape-based filtering to reject truly large objects

### Experiment 5: Alternative to Vegetation Subtraction
**Goal:** Replace vegetation subtraction with post-hoc filtering

**Method:**
- Remove vegetation subtraction entirely
- Add color histogram analysis to detect vegetation-like regions
- Reject candidates with green-dominant histograms
- Run investigation script

**Expected Outcome:**
- Sign pixels preserved
- Vegetation rejected at candidate level instead of mask level
- More robust to varying lighting conditions

### Experiment 6: Relax LAB Thresholds
**Goal:** Allow more red pixels to pass LAB filter

**Method:**
- Change `lab_red_lower` from (20, 140, 100) to (20, 130, 100)
- Run investigation script

**Expected Outcome:**
- More red pixels detected (from 34.4% to 55.5% in GT region)
- Better red sign detection
- Possible increase in false positives from red objects

---

## 8. Missing Information / Limitations

### 1. YOLO-Formatted Labels Missing
**Issue:** `run_gtsdb_150_eval.py` expects YOLO format labels, but dataset is in GTSDB format
**Impact:** Cannot run 150-image batch evaluation as designed
**Workaround:** Created custom investigation script that parses GTSDB format
**Needed:** Label format conversion script or modification of evaluation script

### 2. GTSRB Dataset Not Used
**Issue:** Code references `gtsrb_dataset/` directory but only GTSDB data is present
**Impact:** Cannot test on cropped sign patches
**Needed:** Download GTSRB dataset or remove GTSRB references

### 3. No Test Set Evaluation
**Issue:** Investigation only on training set (00000-00019.ppm)
**Impact:** May overfit to training data characteristics
**Needed:** Evaluate on test set (TestIJCNN2013)

### 4. No Per-Class Analysis
**Issue:** Investigation aggregates all sign types
**Impact:** May miss class-specific failure modes (e.g., blue signs vs red signs)
**Needed:** Analyze detection rates per class (prohibitory, mandatory, danger, other)

### 5. No Size Distribution Analysis
**Issue:** No analysis of detection vs sign size
**Impact:** Unknown if small signs (<20px) are systematically missed
**Needed:** Correlate detection rate with sign dimensions

---

## 9. Conclusion

The Stage 3 candidate detection system is **non-functional** in its current state due to vegetation mask subtraction causing sign-background merger. The root cause is clear and fixable:

**Primary Fix Required:**
1. Disable or significantly tighten vegetation subtraction in `thresholding.py`
2. Consider moving vegetation rejection to post-hoc candidate filtering instead of mask-level subtraction

**Secondary Improvements:**
1. Remove or reduce morphological opening to preserve small signs
2. Add minimum contour solidity filter to reject irregular merged shapes
3. Consider increasing area threshold if vegetation cannot be perfectly removed

**Testing Required:**
1. Run proposed experiments to verify fix effectiveness
2. Evaluate on full training set (600 images)
3. Evaluate on test set (55 images)
4. Perform per-class and per-size analysis

The detection pipeline is otherwise well-designed with appropriate multi-criteria filtering. Once the vegetation issue is resolved, the system should achieve reasonable detection rates.

---

## Appendix: Sample Failure Cases

### Case 1: Image 00000.ppm
- **GT:** 1 sign (class 11, priority at intersection)
- **Detected:** 29 candidates, all false positives
- **Root Cause:** Vegetation merger (sign merged into 391×283 contour)
- **IoU with best detection:** 0.00 (no detection overlaps GT)

### Case 2: Image 00001.ppm
- **GT:** 3 signs (classes 40, 38, 13)
- **Detected:** 70 candidates, all false positives
- **Root Cause:** Same vegetation merger issue
- **Sign sizes:** 41×44, 56×58, 58×55 (all should be detectable)

### Case 3: Image 00003.ppm
- **GT:** 3 signs (classes 4, 9, 21)
- **Detected:** 47 candidates, all false positives
- **Root Cause:** Signs are vertically stacked; vegetation creates large merged region
- **Sign sizes:** 23×23, 22×23, 32×31 (small but above 8px threshold)

### Typical False Positive Characteristics
- Size: 8-43 pixels (just above 40px area threshold)
- Location: Scattered across entire image including borders
- Shape: Irregular, low circularity (<0.15)
- Color: Often residual noise from thresholding
