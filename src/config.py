"""Configuration settings for Stage 1 (Enhancement) and Stage 2 (Color Segmentation)."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Stage1Config:
    """Parameters for Stage 1: Image Enhancement."""
    # Denoising: 'adaptive' selects gentle parameters based on estimated noise to avoid blur
    denoise_method: str = "adaptive"  # 'adaptive', 'bilateral', or 'nlm'
    bilateral_d: int = 5
    bilateral_sigma_color: float = 25.0
    bilateral_sigma_space: float = 10.0
    nlm_h: float = 6.0
    nlm_h_color: float = 6.0

    # Contrast Enhancement (CLAHE in CIELAB color space + normalization)
    clahe_clip_limit: float = 2.5
    clahe_grid_size: Tuple[int, int] = (8, 8)
    normalize_range: bool = True

    # Exposure adjustment & edge sharpening
    gamma: float = 1.15
    sharpen_amount: float = 0.35  # Unsharp masking strength to ensure crisp text & borders


@dataclass
class Stage2Config:
    """Parameters for Stage 2: Color Segmentation and Sign Candidate Verification.
    
    Thresholds calibrated from actual GTSDB pixel measurements:
    - Blue signs: HSV H~100-115, S can be as low as 20 (distant/hazy signs), V wide
    - Red signs: HSV H in [0,10] or [160,180], S typically 50+
    - Yellow signs: HSV H~15-35, S >= 50
    """

    # ======================== HSV Color Ranges ========================
    # OpenCV scale: H: 0-180, S: 0-255, V: 0-255

    # Saturated Red (stop, prohibitory, danger signs)
    # S >= 50 to catch slightly desaturated distant signs while still rejecting gray
    hsv_red_lower1: Tuple[int, int, int] = (0, 50, 40)
    hsv_red_upper1: Tuple[int, int, int] = (12, 255, 255)
    hsv_red_lower2: Tuple[int, int, int] = (155, 50, 40)
    hsv_red_upper2: Tuple[int, int, int] = (180, 255, 255)

    # Blue (Mandatory / direction signs)
    # S >= 40 because real blue signs have S down to ~40 when distant
    hsv_blue_lower: Tuple[int, int, int] = (95, 40, 40)
    hsv_blue_upper: Tuple[int, int, int] = (135, 255, 255)

    # Yellow / Amber (Warning / Caution signs)
    hsv_yellow_lower: Tuple[int, int, int] = (14, 50, 50)
    hsv_yellow_upper: Tuple[int, int, int] = (38, 255, 255)

    # White (Sign interior / numerals — used only for structural verification, NOT combined)
    hsv_white_lower: Tuple[int, int, int] = (0, 0, 170)
    hsv_white_upper: Tuple[int, int, int] = (180, 40, 255)

    # ======================== CIELAB Thresholds ========================
    # L: 0-255, a: 0-255, b: 0-255 (128 is neutral for a* and b*)

    # Red in LAB: a* >= 140 ensures distinct red chrominance
    lab_red_lower: Tuple[int, int, int] = (20, 140, 100)
    lab_red_upper: Tuple[int, int, int] = (255, 255, 255)

    # Blue in LAB: b* <= 115 (blue = low b*)
    # Measured from actual blue sign pixels: b ranges 88-124
    lab_blue_lower: Tuple[int, int, int] = (20, 60, 20)
    lab_blue_upper: Tuple[int, int, int] = (230, 145, 115)

    # Yellow in LAB: b* >= 140
    lab_yellow_lower: Tuple[int, int, int] = (40, 100, 140)
    lab_yellow_upper: Tuple[int, int, int] = (255, 160, 255)

    # ======================== Morphological Operations ========================
    morph_kernel_size: int = 5
    morph_close_iterations: int = 3
    morph_open_iterations: int = 1

    # ======================== Candidate Shape & Geometry Filtering ========================
    min_area: int = 40                # Minimum contour area (lowered for small distant signs)
    max_area: int = 18000             # Maximum contour area (rejects houses, sky patches)
    min_box_dim: int = 8              # Minimum bounding box side length (small distant signs)
    max_box_dim: int = 180            # Maximum bounding box side — signs are compact
    min_aspect_ratio: float = 0.35    # Signs are roughly square/circular/triangular
    max_aspect_ratio: float = 2.0     # Some direction signs are wider

    # Circularity: 4*pi*area / perimeter^2 (circle=1.0, irregular shape~0)
    min_circularity: float = 0.15     # Relaxed for triangular and rectangular signs
    # Convexity: contour area / convex hull area
    min_convexity: float = 0.40       # Signs are convex shapes

    # ======================== Spatial Filters ========================
    max_road_y_ratio: float = 0.88    # Ignore reflections in bottom road asphalt
    min_sky_y_ratio: float = 0.02     # Ignore artifacts at very top edge of frame

    # ======================== Color Density Verification ========================
    min_color_density: float = 0.05   # At least 5% of bbox must be sign-colored
    max_color_density: float = 0.95   # If >95%, probably a solid color surface

    # ======================== Edge-Based Verification ========================
    min_edge_density: float = 0.01    # Minimum edge pixels in bbox region
    max_edge_density: float = 0.65    # Too many edges = textured background

    # ======================== Structural Verification ========================
    require_sign_structure: bool = True


@dataclass
class PipelineConfig:
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
