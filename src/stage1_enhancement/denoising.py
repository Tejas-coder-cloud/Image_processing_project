"""Edge-preserving denoising and detail enhancement (Stage 1)."""

import cv2
import numpy as np


def denoise_bilateral(
    image_bgr: np.ndarray,
    d: int = 5,
    sigma_color: float = 25.0,
    sigma_space: float = 10.0
) -> np.ndarray:
    """Edge-preserving bilateral smoothing.

    Keeps spatial sigma small (e.g., 10-15) to prevent blurring sign borders,
    numerals, and fine road details.
    """
    return cv2.bilateralFilter(
        image_bgr,
        d=d,
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space)
    )


def denoise_nlm(
    image_bgr: np.ndarray,
    h: float = 6.0,
    h_color: float = 6.0,
    template_window_size: int = 5,
    search_window_size: int = 15
) -> np.ndarray:
    """Non-Local Means (NLM) color denoising with restrained patch radii."""
    return cv2.fastNlMeansDenoisingColored(
        image_bgr,
        None,
        h=float(h),
        hColor=float(h_color),
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size
    )


def denoise_adaptive(
    image_bgr: np.ndarray,
    noise_sigma: float
) -> np.ndarray:
    """Adaptively scale denoising strength based on estimated Immerkaer noise variance.

    If noise is very low (sigma < 1.5), applies minimal smoothing to avoid blur.
    """
    if noise_sigma < 1.5:
        # Image is already clean: very gentle filter
        return denoise_bilateral(image_bgr, d=3, sigma_color=15.0, sigma_space=6.0)
    elif noise_sigma < 5.0:
        # Mild sensor noise
        return denoise_bilateral(image_bgr, d=5, sigma_color=25.0, sigma_space=10.0)
    else:
        # Moderate to severe noise
        return denoise_bilateral(image_bgr, d=7, sigma_color=35.0, sigma_space=14.0)


def apply_denoising(
    image_bgr: np.ndarray,
    method: str = "adaptive",
    noise_sigma: float = 2.0,
    d: int = 5,
    sigma_color: float = 25.0,
    sigma_space: float = 10.0,
    nlm_h: float = 6.0,
    nlm_h_color: float = 6.0
) -> np.ndarray:
    """Apply the chosen denoising algorithm."""
    m = method.lower()
    if m == "nlm":
        return denoise_nlm(image_bgr, h=nlm_h, h_color=nlm_h_color)
    elif m == "adaptive":
        return denoise_adaptive(image_bgr, noise_sigma=noise_sigma)
    else:
        return denoise_bilateral(image_bgr, d=d, sigma_color=sigma_color, sigma_space=sigma_space)
