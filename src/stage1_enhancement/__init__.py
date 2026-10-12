"""Stage 1: Image Enhancement Package."""

from .noise_analysis import (
    compute_power_spectrum,
    estimate_noise_variance_immerkaer,
    assess_quality,
)
from .denoising import (
    denoise_bilateral,
    denoise_nlm,
    denoise_adaptive,
    apply_denoising,
)
from .contrast import (
    apply_clahe_lab,
    apply_gamma_correction,
    apply_unsharp_mask,
    enhance_contrast_and_lighting,
    compute_channel_histograms,
)

__all__ = [
    "compute_power_spectrum",
    "estimate_noise_variance_immerkaer",
    "assess_quality",
    "denoise_bilateral",
    "denoise_nlm",
    "denoise_adaptive",
    "apply_denoising",
    "apply_clahe_lab",
    "apply_gamma_correction",
    "apply_unsharp_mask",
    "enhance_contrast_and_lighting",
    "compute_channel_histograms",
]
