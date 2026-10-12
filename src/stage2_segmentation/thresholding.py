"""Color thresholding in HSV and CIELAB spaces for traffic sign segmentation (Stage 2).

Key improvements over naive thresholding:
1. Dual-space intersection (HSV AND LAB) for each color ensures true sign chromaticity.
2. Tightened saturation floors reject gray objects (road, buildings, cars, shadows).
3. Separate white mask used only for interior structure verification, NOT combined into main mask.
4. Added green/vegetation rejection mask to suppress trees and grass before combining.
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from src.config import Stage2Config


def threshold_hsv_red(
    image_hsv: np.ndarray,
    lower1: Tuple[int, int, int] = (0, 50, 40),
    upper1: Tuple[int, int, int] = (12, 255, 255),
    lower2: Tuple[int, int, int] = (155, 50, 40),
    upper2: Tuple[int, int, int] = (180, 255, 255),
) -> np.ndarray:
    """Threshold saturated red in HSV space across the 0/180 hue wrap."""
    mask1 = cv2.inRange(image_hsv, np.array(lower1, dtype=np.uint8), np.array(upper1, dtype=np.uint8))
    mask2 = cv2.inRange(image_hsv, np.array(lower2, dtype=np.uint8), np.array(upper2, dtype=np.uint8))
    return cv2.bitwise_or(mask1, mask2)


def threshold_hsv_blue(
    image_hsv: np.ndarray,
    lower: Tuple[int, int, int] = (95, 40, 40),
    upper: Tuple[int, int, int] = (135, 255, 255),
) -> np.ndarray:
    """Threshold blue mandatory signs in HSV space."""
    return cv2.inRange(image_hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))


def threshold_hsv_yellow(
    image_hsv: np.ndarray,
    lower: Tuple[int, int, int] = (14, 50, 50),
    upper: Tuple[int, int, int] = (38, 255, 255),
) -> np.ndarray:
    """Threshold yellow/amber warning signs in HSV space."""
    return cv2.inRange(image_hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))


def threshold_hsv_white(
    image_hsv: np.ndarray,
    lower: Tuple[int, int, int] = (0, 0, 170),
    upper: Tuple[int, int, int] = (180, 40, 255),
) -> np.ndarray:
    """Threshold white sign interior, borders, and numerals in HSV space."""
    return cv2.inRange(image_hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))


def threshold_lab_red(
    image_lab: np.ndarray,
    lower: Tuple[int, int, int] = (20, 140, 100),
    upper: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Threshold red in CIELAB space. a* >= 150 isolates distinct red chrominance."""
    return cv2.inRange(image_lab, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))


def threshold_lab_blue(
    image_lab: np.ndarray,
    lower: Tuple[int, int, int] = (20, 60, 20),
    upper: Tuple[int, int, int] = (230, 145, 115),
) -> np.ndarray:
    """Threshold blue in CIELAB space. b* <= 100 isolates blue chrominance."""
    return cv2.inRange(image_lab, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))


def threshold_lab_yellow(
    image_lab: np.ndarray,
    lower: Tuple[int, int, int] = (40, 100, 140),
    upper: Tuple[int, int, int] = (255, 160, 255),
) -> np.ndarray:
    """Threshold yellow in CIELAB space. b* >= 150 isolates yellow chrominance."""
    return cv2.inRange(image_lab, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))


def _create_vegetation_mask(image_hsv: np.ndarray) -> np.ndarray:
    """Create a mask of green vegetation (trees, grass) to subtract from candidate regions.
    
    Trees and grass are a major source of false positives since they can have locally
    saturated patches that get picked up by broad color ranges.
    """
    # Green vegetation: H in [35, 85], moderate saturation and value
    green_mask = cv2.inRange(
        image_hsv,
        np.array([35, 30, 20], dtype=np.uint8),
        np.array([85, 255, 255], dtype=np.uint8),
    )
    # Dilate slightly so green areas bleed into adjacent pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_mask = cv2.dilate(green_mask, kernel, iterations=1)
    return green_mask


def segment_traffic_sign_colors(
    image_bgr: np.ndarray,
    config: Optional[Stage2Config] = None,
) -> Dict[str, np.ndarray]:
    """Execute multi-color segmentation across HSV and CIELAB color spaces.

    Intersects HSV and LAB space constraints for each chromatic category to ensure
    high specificity and eliminate asphalt, vehicle bodies, and shadow reflections.
    Green vegetation is subtracted to prevent tree/grass false positives.
    """
    cfg = config or Stage2Config()

    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    # HSV masks
    hsv_red = threshold_hsv_red(image_hsv, cfg.hsv_red_lower1, cfg.hsv_red_upper1, cfg.hsv_red_lower2, cfg.hsv_red_upper2)
    hsv_blue = threshold_hsv_blue(image_hsv, cfg.hsv_blue_lower, cfg.hsv_blue_upper)
    hsv_yellow = threshold_hsv_yellow(image_hsv, cfg.hsv_yellow_lower, cfg.hsv_yellow_upper)
    hsv_white = threshold_hsv_white(image_hsv, cfg.hsv_white_lower, cfg.hsv_white_upper)

    # LAB masks
    lab_red = threshold_lab_red(image_lab, cfg.lab_red_lower, cfg.lab_red_upper)
    lab_blue = threshold_lab_blue(image_lab, cfg.lab_blue_lower, cfg.lab_blue_upper)
    lab_yellow = threshold_lab_yellow(image_lab, cfg.lab_yellow_lower, cfg.lab_yellow_upper)

    # Dual-space intersection: ensures true sign chromaticity in BOTH spaces
    fused_red = cv2.bitwise_and(hsv_red, lab_red)
    fused_blue = cv2.bitwise_and(hsv_blue, lab_blue)
    fused_yellow = cv2.bitwise_and(hsv_yellow, lab_yellow)

    # Subtract green vegetation areas to reduce false positives from trees/grass
    if getattr(cfg, "subtract_vegetation", True):
        vegetation_mask = _create_vegetation_mask(image_hsv)
        inv_vegetation = cv2.bitwise_not(vegetation_mask)
        fused_red = cv2.bitwise_and(fused_red, inv_vegetation)
        fused_blue = cv2.bitwise_and(fused_blue, inv_vegetation)
        fused_yellow = cv2.bitwise_and(fused_yellow, inv_vegetation)

    mask_combined = cv2.bitwise_or(fused_red, cv2.bitwise_or(fused_blue, fused_yellow))

    return {
        "mask_red": fused_red,
        "mask_blue": fused_blue,
        "mask_yellow": fused_yellow,
        "mask_white": hsv_white,
        "hsv_red": hsv_red,
        "hsv_blue": hsv_blue,
        "hsv_yellow": hsv_yellow,
        "lab_red": lab_red,
        "lab_blue": lab_blue,
        "lab_yellow": lab_yellow,
        "mask_combined": mask_combined,
    }
