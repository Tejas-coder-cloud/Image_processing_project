"""Visualization and report generator for Stage 1 and Stage 2 pipeline results."""

from pathlib import Path
from typing import Any, Dict
import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_stage1_report(stage1_res: Dict[str, Any], save_path: Path) -> None:
    """Generate visual report for Stage 1: Image Enhancement."""
    raw_bgr = stage1_res["raw_image"]
    denoised_bgr = stage1_res["denoised_image"]
    enhanced_bgr = stage1_res["enhanced_image"]
    q_raw = stage1_res["quality_raw"]
    q_enh = stage1_res["quality_enhanced"]

    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0F172A")
    plt.rcParams["text.color"] = "#E2E8F0"
    plt.rcParams["axes.labelcolor"] = "#E2E8F0"
    plt.rcParams["xtick.color"] = "#94A3B8"
    plt.rcParams["ytick.color"] = "#94A3B8"

    # Row 1, Col 1: Raw Input
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(raw_rgb)
    ax1.set_title(f"Input Road Image\n(Noise Sigma: {q_raw['noise_sigma']:.2f})", fontsize=11, color="white")
    ax1.axis("off")

    # Row 1, Col 2: Power Spectrum
    ax2 = plt.subplot(2, 3, 2)
    freqs = stage1_res["radial_frequencies"]
    power = stage1_res["radial_power"]
    ax2.set_facecolor("#1E293B")
    ax2.plot(freqs, power, color="#38BDF8", linewidth=2.0, label="Radial Power")
    ax2.fill_between(freqs, power, color="#38BDF8", alpha=0.25)
    ax2.set_title(f"Noise & Power Spectrum Analysis\nBlur Score: {q_raw['blur_laplacian_var']:.1f}", fontsize=11, color="white")
    ax2.set_xlabel("Normalized Spatial Frequency")
    ax2.set_ylabel("Log Power (normalized)")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right", facecolor="#334155", edgecolor="none")

    # Row 1, Col 3: Denoised (Edge-Preserving)
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(denoised_rgb)
    ax3.set_title("Denoised Image\n(Edge-Preserving Filter)", fontsize=11, color="white")
    ax3.axis("off")

    # Row 2, Col 1: Raw Histogram
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_facecolor("#1E293B")
    hist_raw = stage1_res["hist_raw"]
    ax4.plot(hist_raw["red"], color="#EF4444", label="R", alpha=0.8)
    ax4.plot(hist_raw["green"], color="#10B981", label="G", alpha=0.8)
    ax4.plot(hist_raw["blue"], color="#3B82F6", label="B", alpha=0.8)
    ax4.plot(hist_raw["gray"], color="#F8FAFC", label="Luminance", linewidth=1.5, linestyle="--")
    ax4.set_title("Before: Color & Luminance Histogram", fontsize=11, color="white")
    ax4.set_xlim([0, 255])
    ax4.grid(True, linestyle="--", alpha=0.3)
    ax4.legend(loc="upper right", facecolor="#334155", edgecolor="none")

    # Row 2, Col 2: Enhanced Histogram
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_facecolor("#1E293B")
    hist_enh = stage1_res["hist_enhanced"]
    ax5.plot(hist_enh["red"], color="#EF4444", label="R", alpha=0.8)
    ax5.plot(hist_enh["green"], color="#10B981", label="G", alpha=0.8)
    ax5.plot(hist_enh["blue"], color="#3B82F6", label="B", alpha=0.8)
    ax5.plot(hist_enh["gray"], color="#F8FAFC", label="Luminance", linewidth=1.5, linestyle="--")
    ax5.set_title(f"After: CLAHE (LAB) + Gamma\n(Contrast RMS: {q_enh['rms_contrast']:.1f})", fontsize=11, color="white")
    ax5.set_xlim([0, 255])
    ax5.grid(True, linestyle="--", alpha=0.3)
    ax5.legend(loc="upper right", facecolor="#334155", edgecolor="none")

    # Row 2, Col 3: Final Enhanced Result
    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(enhanced_rgb)
    ax6.set_title("Stage 1 Output: Enhanced Image\n(Ready for Color Segmentation)", fontsize=11, color="#38BDF8")
    ax6.axis("off")

    plt.suptitle("STAGE 1 — IMAGE ENHANCEMENT REPORT", fontsize=16, fontweight="bold", color="#F8FAFC", y=0.98)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def plot_stage2_report(stage2_res: Dict[str, Any], save_path: Path) -> None:
    """Generate visual report for Stage 2: Color Segmentation."""
    rgb = stage2_res["rgb"]
    red_mask = stage2_res["cleaned_red"]
    blue_mask = stage2_res["cleaned_blue"]
    yellow_mask = stage2_res["cleaned_yellow"]
    final_mask = stage2_res["binary_mask_cleaned"]
    overlay = cv2.cvtColor(stage2_res["overlay"], cv2.COLOR_BGR2RGB)
    bboxes = stage2_res["bounding_boxes"]

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0F172A")
    plt.rcParams["text.color"] = "#E2E8F0"

    # Row 1, Col 1: Enhanced Input
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(rgb)
    ax1.set_title("Input Enhanced Image", fontsize=11, color="white")
    ax1.axis("off")

    # Row 1, Col 2: Red Mask (HSV + LAB)
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(red_mask, cmap="gray")
    ax2.set_title("Red Threshold (Stop / Danger / Prohibitory)\nHSV [0,10]+[160,180] & LAB high a*", fontsize=10, color="#F87171")
    ax2.axis("off")

    # Row 1, Col 3: Blue Mask (HSV + LAB)
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(blue_mask, cmap="gray")
    ax3.set_title("Blue Threshold (Mandatory / Directions)\nHSV [100,130] & LAB low b*", fontsize=10, color="#60A5FA")
    ax3.axis("off")

    # Row 2, Col 1: Yellow Mask (HSV + LAB)
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(yellow_mask, cmap="gray")
    ax4.set_title("Yellow Threshold (Warning / Caution)\nHSV [15,38] & LAB high b*", fontsize=10, color="#FBBF24")
    ax4.axis("off")

    # Row 2, Col 2: Final Fused & Morphologically Cleaned Mask
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(final_mask, cmap="gray")
    ax5.set_title("Final Binary Mask\n(Fused Multi-Color + Morphology)", fontsize=11, color="white")
    ax5.axis("off")

    # Row 2, Col 3: Detected Candidate Sign Regions Overlay
    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(overlay)
    ax6.set_title(f"Segmented Candidate Regions\n({len(bboxes)} sign candidates localized)", fontsize=11, color="#34D399")
    ax6.axis("off")

    plt.suptitle("STAGE 2 — COLOR SEGMENTATION REPORT", fontsize=16, fontweight="bold", color="#F8FAFC", y=0.98)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def plot_pipeline_summary(stage1_res: Dict[str, Any], stage2_res: Dict[str, Any], save_path: Path) -> None:
    """Generate high-level 4-panel progression summary from raw image to segmented candidates."""
    raw_rgb = cv2.cvtColor(stage1_res["raw_image"], cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(stage1_res["enhanced_image"], cv2.COLOR_BGR2RGB)
    binary_mask = stage2_res["binary_mask_cleaned"]
    overlay = cv2.cvtColor(stage2_res["overlay"], cv2.COLOR_BGR2RGB)
    num_candidates = len(stage2_res["bounding_boxes"])

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#0F172A")
    plt.rcParams["text.color"] = "#E2E8F0"

    panels = [
        (axes[0], raw_rgb, "1. Input Road Scene", "Raw Image (GTSDB)"),
        (axes[1], enhanced_rgb, "2. Stage 1: Enhanced", "Denoising + CLAHE (LAB) + Gamma"),
        (axes[2], binary_mask, "3. Stage 2: Binary Mask", "HSV + LAB Fused Thresholds"),
        (axes[3], overlay, "4. Sign Candidates Overlay", f"{num_candidates} Candidate Bounding Boxes"),
    ]

    for ax, img, title, subtitle in panels:
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11, color="white")
        ax.axis("off")

    plt.suptitle("TRAFFIC SIGN IMAGE PROCESSING PIPELINE: STAGE 1 & STAGE 2", fontsize=15, fontweight="bold", color="#38BDF8", y=1.03)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
