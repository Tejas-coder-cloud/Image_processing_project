"""Stage 2: Color Segmentation Package."""

from .color_spaces import convert_color_spaces, get_channel_breakdowns
from .thresholding import (
    threshold_hsv_red,
    threshold_hsv_blue,
    threshold_hsv_yellow,
    threshold_hsv_white,
    threshold_lab_red,
    threshold_lab_blue,
    threshold_lab_yellow,
    segment_traffic_sign_colors,
)
from .morphology import (
    apply_morphology,
    extract_candidate_regions,
    draw_segmentation_overlay,
    non_max_suppression_boxes,
)

__all__ = [
    "convert_color_spaces",
    "get_channel_breakdowns",
    "threshold_hsv_red",
    "threshold_hsv_blue",
    "threshold_hsv_yellow",
    "threshold_hsv_white",
    "threshold_lab_red",
    "threshold_lab_blue",
    "threshold_lab_yellow",
    "segment_traffic_sign_colors",
    "apply_morphology",
    "extract_candidate_regions",
    "draw_segmentation_overlay",
    "non_max_suppression_boxes",
]
