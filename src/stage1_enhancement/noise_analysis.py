"""Noise estimation, power spectrum analysis, and image quality assessment (Stage 1)."""

from typing import Dict, Tuple
import cv2
import numpy as np


def compute_power_spectrum(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D Log Power Spectrum and 1D Radially Averaged Power Spectrum.

    Args:
        image_bgr: Input BGR image.

    Returns:
        (power_spectrum_2d_log, radial_frequencies, radial_power):
            - 2D log power spectrum array (for display)
            - 1D normalized spatial frequency axis
            - 1D radial power spectrum curve
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # Apply Hanning window to reduce edge boundary leakage
    window_y = np.hanning(h)
    window_x = np.hanning(w)
    window_2d = np.outer(window_y, window_x)
    windowed_gray = (gray - np.mean(gray)) * window_2d

    # 2D FFT and shifted zero-frequency to center
    f_transform = np.fft.fft2(windowed_gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    power_2d = magnitude ** 2
    power_2d_log = np.log1p(power_2d)

    # Radially averaged power spectrum (1D)
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radii = np.hypot(x - center_x, y - center_y).astype(int)

    max_radius = min(center_x, center_y)
    radial_sum = np.bincount(radii.ravel(), weights=power_2d.ravel())
    radial_count = np.bincount(radii.ravel())

    # Avoid zero division
    valid = radial_count[:max_radius] > 0
    radial_power = np.zeros(max_radius, dtype=np.float64)
    radial_power[valid] = radial_sum[:max_radius][valid] / radial_count[:max_radius][valid]

    # Normalize radial curve for plotting
    radial_frequencies = np.linspace(0.0, 1.0, max_radius)
    if np.max(radial_power) > 0:
        radial_power_log = np.log1p(radial_power)
        radial_power_norm = (radial_power_log - np.min(radial_power_log)) / (
            np.max(radial_power_log) - np.min(radial_power_log) + 1e-8
        )
    else:
        radial_power_norm = radial_power

    return power_2d_log, radial_frequencies, radial_power_norm


def estimate_noise_variance_immerkaer(image_bgr: np.ndarray) -> float:
    """Estimate additive Gaussian noise standard deviation using Immerkaer's fast Laplacian method.

    Reference:
        J. Immerkaer, "Fast Noise Variance Estimation", Computer Vision and Image
        Understanding, Vol. 64, No. 2, 1996, pp. 300-302.

    Formula:
        sigma = sqrt(pi / 2) * (1 / (6 * (W - 2) * (H - 2))) * sum(|I * M|)
        where M is the 3x3 Laplacian mask:
        [[ 1, -2,  1],
         [-2,  4, -2],
         [ 1, -2,  1]]
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape

    if h < 3 or w < 3:
        return 0.0

    mask = np.array([[1, -2, 1],
                     [-2, 4, -2],
                     [1, -2, 1]], dtype=np.float64)

    convolved = cv2.filter2D(gray, -1, mask, borderType=cv2.BORDER_CONSTANT)
    # Exclude borders where mask is invalid
    valid_region = convolved[1:-1, 1:-1]

    sum_abs = np.sum(np.abs(valid_region))
    sigma = np.sqrt(np.pi / 2.0) * (1.0 / (6.0 * (w - 2) * (h - 2))) * sum_abs
    return float(sigma)


def assess_quality(image_bgr: np.ndarray) -> Dict[str, float]:
    """Perform comprehensive image quality assessment before Stage 1 enhancement.

    Metrics:
        - noise_sigma: Immerkaer noise estimate
        - blur_metric: Variance of Laplacian (higher = sharper, lower = blurrier)
        - dynamic_range: (Max - Min) grayscale intensity
        - mean_brightness: Average luminance level
        - rms_contrast: Standard deviation of pixel intensities
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    noise_sigma = estimate_noise_variance_immerkaer(image_bgr)
    blur_metric = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_val = float(np.mean(gray))
    rms_contrast = float(np.std(gray))
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)

    # Classify noise level (0 = Clean, 100 = Severe Noise)
    # Typical sigma: clean < 2.0, mild 2-6, moderate 6-12, severe > 12
    noise_index = min(100.0, (noise_sigma / 15.0) * 100.0)

    return {
        "noise_sigma": round(noise_sigma, 3),
        "noise_index": round(noise_index, 1),
        "blur_laplacian_var": round(blur_metric, 2),
        "mean_brightness": round(mean_val, 2),
        "rms_contrast": round(rms_contrast, 2),
        "dynamic_range": int(max_val - min_val),
    }
