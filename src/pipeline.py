"""Unified pipeline runner coordinating Stage 1 (Enhancement) and Stage 2 (Color Segmentation)."""

from typing import Any, Dict, Optional
import cv2
import numpy as np

from src.config import PipelineConfig, Stage1Config, Stage2Config
from src.stage1_enhancement import (
    assess_quality,
    compute_power_spectrum,
    apply_denoising,
    enhance_contrast_and_lighting,
    compute_channel_histograms,
)
from src.stage2_segmentation import (
    convert_color_spaces,
    get_channel_breakdowns,
    segment_traffic_sign_colors,
    apply_morphology,
    extract_candidate_regions,
    draw_segmentation_overlay,
)


def run_stage1(image_bgr: np.ndarray, config: Optional[Stage1Config] = None) -> Dict[str, Any]:
    """Execute Stage 1: Image Enhancement.

    Steps:
        1. Noise estimation & quality metrics (Immerkaer formula)
        2. 2D & 1D Fourier Power Spectrum analysis
        3. Adaptive edge-preserving denoising (prevents blur)
        4. CLAHE contrast expansion in CIELAB + Gamma correction
        5. Detail and edge unsharp sharpening
        6. Before & After histogram analysis
    """
    cfg = config or Stage1Config()

    # Step 1: Quality assessment on raw image
    quality_raw = assess_quality(image_bgr)

    # Step 2: Power spectrum analysis
    ps_2d_log, radial_freqs, radial_power = compute_power_spectrum(image_bgr)

    # Step 3: Denoising (adaptive to avoid blurring clean images)
    denoised_bgr = apply_denoising(
        image_bgr,
        method=cfg.denoise_method,
        noise_sigma=quality_raw["noise_sigma"],
        d=cfg.bilateral_d,
        sigma_color=cfg.bilateral_sigma_color,
        sigma_space=cfg.bilateral_sigma_space,
        nlm_h=cfg.nlm_h,
        nlm_h_color=cfg.nlm_h_color,
    )

    # Step 4 & 5: CLAHE in CIELAB, dynamic range normalization, and unsharp sharpening
    enhanced_bgr = enhance_contrast_and_lighting(
        denoised_bgr,
        clahe_clip_limit=cfg.clahe_clip_limit,
        clahe_grid_size=cfg.clahe_grid_size,
        gamma=cfg.gamma,
        sharpen_amount=cfg.sharpen_amount,
    )

    # Step 6: Metrics and histograms on enhanced image
    quality_enhanced = assess_quality(enhanced_bgr)
    hist_raw = compute_channel_histograms(image_bgr)
    hist_enhanced = compute_channel_histograms(enhanced_bgr)

    return {
        "raw_image": image_bgr,
        "denoised_image": denoised_bgr,
        "enhanced_image": enhanced_bgr,
        "quality_raw": quality_raw,
        "quality_enhanced": quality_enhanced,
        "power_spectrum_2d": ps_2d_log,
        "radial_frequencies": radial_freqs,
        "radial_power": radial_power,
        "hist_raw": hist_raw,
        "hist_enhanced": hist_enhanced,
    }


def run_stage2(enhanced_bgr: np.ndarray, config: Optional[Stage2Config] = None) -> Dict[str, Any]:
    """Execute Stage 2: Color Segmentation.

    Steps:
        1. RGB -> HSV & CIELAB conversions
        2. Channel extractions (H, S, V, L, a, b)
        3. Dual-space saturated color thresholding (Red, Blue, Yellow, White)
           with vegetation subtraction
        4. Morphological closing (bridges sign borders) and opening (removes noise)
        5. Multi-criteria structural sign verification and false positive rejection:
           - Circularity, convexity, color density, edge density
           - Traffic light detection & rejection
           - Large uniform region rejection
           - Interior structure verification
        6. Candidate sign bounding box extraction & overlay
    """
    cfg = config or Stage2Config()
    h, w, _ = enhanced_bgr.shape

    # Step 1 & 2: Color space conversions & channels
    img_rgb, img_hsv, img_lab = convert_color_spaces(enhanced_bgr)
    channels = get_channel_breakdowns(enhanced_bgr)

    # Step 3: Multi-color segmentation with vegetation subtraction
    color_masks = segment_traffic_sign_colors(enhanced_bgr, cfg)

    # Step 4: Morphological post-processing (closing first bridges circular border rings)
    raw_combined_mask = color_masks["mask_combined"]
    cleaned_mask = apply_morphology(
        raw_combined_mask,
        kernel_size=cfg.morph_kernel_size,
        close_iters=cfg.morph_close_iterations,
        open_iters=cfg.morph_open_iterations,
    )

    cleaned_red = apply_morphology(color_masks["mask_red"], cfg.morph_kernel_size, cfg.morph_close_iterations, cfg.morph_open_iterations)
    cleaned_blue = apply_morphology(color_masks["mask_blue"], cfg.morph_kernel_size, cfg.morph_close_iterations, cfg.morph_open_iterations)
    cleaned_yellow = apply_morphology(color_masks["mask_yellow"], cfg.morph_kernel_size, cfg.morph_close_iterations, cfg.morph_open_iterations)

    # Step 5: Multi-criteria bounding box detection with structural verification
    # Pass enhanced_bgr for edge-based and texture-based analysis
    bboxes, contours, verified_mask = extract_candidate_regions(
        cleaned_mask,
        config=cfg,
        aux_masks=color_masks,
        image_shape=(h, w),
        enhanced_bgr=enhanced_bgr,
    )
    # Overlay highlights only the verified traffic signs
    overlay = draw_segmentation_overlay(enhanced_bgr, verified_mask, bboxes)

    return {
        "rgb": img_rgb,
        "hsv": img_hsv,
        "lab": img_lab,
        "channels": channels,
        "color_masks_raw": color_masks,
        "binary_mask_raw": raw_combined_mask,
        "binary_mask_cleaned": verified_mask,  # Verified mask containing ONLY confirmed traffic signs
        "binary_mask_morph": cleaned_mask,     # Intermediate morphological mask before candidate filter
        "cleaned_red": cleaned_red,
        "cleaned_blue": cleaned_blue,
        "cleaned_yellow": cleaned_yellow,
        "bounding_boxes": bboxes,
        "contours": contours,
        "overlay": overlay,
    }


def run_full_pipeline(image_bgr: np.ndarray, config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    """Execute complete Stage 1 + Stage 2 pipeline."""
    cfg = config or PipelineConfig()
    stage1_res = run_stage1(image_bgr, cfg.stage1)
    stage2_res = run_stage2(stage1_res["enhanced_image"], cfg.stage2)

    return {
        "stage1": stage1_res,
        "stage2": stage2_res,
    }
