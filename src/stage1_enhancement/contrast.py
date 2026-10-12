"""CLAHE in CIELAB, dynamic range expansion, and detail sharpening (Stage 1)."""

from typing import Dict, Tuple
import cv2
import numpy as np


def apply_clahe_lab(
    image_bgr: np.ndarray,
    clip_limit: float = 2.5,
    grid_size: Tuple[int, int] = (8, 8),
    normalize_range: bool = True
) -> np.ndarray:
    """Apply CLAHE to the L* (luminance) channel in CIELAB color space.

    In CIELAB space, modifying the L* channel enhances local contrast without
    introducing chromatic distortion. Normalizing the L* channel spreads out
    the intensity distribution across the dynamic range [15, 240], transforming
    dark/gloomy scenes into clear, high-contrast images.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=grid_size)
    l_enhanced = clahe.apply(l_channel)

    if normalize_range:
        l_enhanced = cv2.normalize(l_enhanced, None, 15, 240, cv2.NORM_MINMAX)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def apply_gamma_correction(image_bgr: np.ndarray, gamma: float = 1.15) -> np.ndarray:
    """Non-linear exposure adjustment using a fast lookup table."""
    if abs(gamma - 1.0) < 1e-4:
        return image_bgr.copy()

    inv_gamma = 1.0 / max(1e-4, gamma)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image_bgr, table)


def apply_unsharp_mask(
    image_bgr: np.ndarray,
    amount: float = 1.35,
    radius: float = 1.2
) -> np.ndarray:
    """Sharpen high-frequency details, text, and sign edges using unsharp masking.

    Formula:
        Sharpened = (1 + amount) * Original - amount * GaussianBlurred
    """
    if amount <= 0:
        return image_bgr.copy()

    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=float(radius))
    sharpened = cv2.addWeighted(image_bgr, 1.0 + amount, blurred, -float(amount), 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_contrast_and_lighting(
    image_bgr: np.ndarray,
    clahe_clip_limit: float = 2.5,
    clahe_grid_size: Tuple[int, int] = (8, 8),
    gamma: float = 1.15,
    sharpen_amount: float = 0.35
) -> np.ndarray:
    """Execute complete contrast enhancement pipeline: CLAHE -> Gamma -> Detail Sharpening."""
    clahe_img = apply_clahe_lab(image_bgr, clip_limit=clahe_clip_limit, grid_size=clahe_grid_size)
    gamma_img = apply_gamma_correction(clahe_img, gamma=gamma)
    if sharpen_amount > 0:
        return apply_unsharp_mask(gamma_img, amount=sharpen_amount, radius=1.2)
    return gamma_img


def compute_channel_histograms(image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute color and luminance histograms for visualization."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()

    hist_b = cv2.calcHist([image_bgr], [0], None, [256], [0, 256]).ravel()
    hist_g = cv2.calcHist([image_bgr], [1], None, [256], [0, 256]).ravel()
    hist_r = cv2.calcHist([image_bgr], [2], None, [256], [0, 256]).ravel()

    return {
        "gray": hist_gray,
        "blue": hist_b,
        "green": hist_g,
        "red": hist_r,
    }
