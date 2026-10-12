"""Color space transformations (RGB -> HSV, RGB -> CIELAB) and channel extraction (Stage 2)."""

from typing import Dict, Tuple
import cv2
import numpy as np


def convert_color_spaces(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert input image into RGB, HSV, and CIELAB color spaces.

    Returns:
        (image_rgb, image_hsv, image_lab)
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    return image_rgb, image_hsv, image_lab


def get_channel_breakdowns(image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract individual channels from HSV and CIELAB for analysis and visualization.

    Returns:
        dict with keys: 'H', 'S', 'V', 'L', 'a', 'b'
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    h, s, v = cv2.split(image_hsv)
    l, a, b = cv2.split(image_lab)

    return {
        "H": h,
        "S": s,
        "V": v,
        "L": l,
        "a": a,
        "b": b,
    }
