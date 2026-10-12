"""Morphological post-processing and candidate traffic sign verification (Stage 2).

This module implements a rigorous multi-criteria filtering pipeline to reject
false positives (vehicles, buildings, traffic lights, trees, road markings)
while retaining genuine traffic signs. The approach uses ONLY classical image
processing techniques — no CNN, no YOLO, no machine learning.

Verification criteria:
1. Area & bounding box dimension limits
2. Aspect ratio (signs are roughly square or slightly rectangular)
3. Circularity: 4*pi*A / P^2 (rejects irregular jagged blobs)
4. Convexity: contour area / convex hull area (signs are convex shapes)
5. Color density: fraction of sign-color pixels within the bounding box
6. Edge density: Canny edge response within the bounding box
7. Spatial position: signs don't appear at very bottom (road) or very top (sky)
8. Size-position relationship: large objects low in frame are vehicles, not signs
9. Traffic light pattern detection and rejection
10. Large uniform/textured region rejection (buildings, sky patches)
11. Vehicle red/taillight rejection (very low in frame + wide aspect)
12. Interior structure verification
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from src.config import Stage2Config


def apply_morphology(
    binary_mask: np.ndarray,
    kernel_size: int = 5,
    close_iters: int = 3,
    open_iters: int = 1
) -> np.ndarray:
    """Apply morphological operations to create solid sign masks and remove noise.

    - Closing (dilation followed by erosion): bridges boundary rings (e.g. speed limit red rings)
      and fills interior gaps.
    - Opening (erosion followed by dilation): eliminates isolated sub-pixel noise.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = binary_mask

    if close_iters > 0:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

    if open_iters > 0:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=open_iters)

    return cleaned


def _compute_circularity(contour: np.ndarray) -> float:
    """Compute circularity = 4*pi*area / perimeter^2. Perfect circle = 1.0."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1e-6:
        return 0.0
    return (4.0 * np.pi * area) / (perimeter * perimeter)


def _compute_convexity(contour: np.ndarray) -> float:
    """Compute convexity = contour area / convex hull area."""
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area < 1e-6:
        return 0.0
    return float(area) / hull_area


def _compute_edge_density(image_gray: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Compute the density of Canny edge pixels within a bounding box region."""
    crop = image_gray[y:y + h, x:x + w]
    if crop.size == 0:
        return 0.0
    edges = cv2.Canny(crop, 50, 150)
    edge_count = cv2.countNonZero(edges)
    total_pixels = crop.shape[0] * crop.shape[1]
    return float(edge_count) / max(1, total_pixels)


def _compute_color_density(mask_crop: np.ndarray, w: int, h: int) -> float:
    """Compute fraction of sign-colored pixels within the bounding box."""
    if mask_crop is None or mask_crop.size == 0:
        return 0.0
    color_count = cv2.countNonZero(mask_crop)
    total_pixels = w * h
    return float(color_count) / max(1, total_pixels)


def _is_likely_vehicle_region(
    x: int, y: int, w: int, h: int,
    img_h: int, img_w: int,
    red_mask: Optional[np.ndarray] = None,
) -> bool:
    """Detect and reject vehicle bodies, taillights, and headlights.

    Vehicles appear in the lower half of the image and tend to be:
    - Large red blobs (truck bodies, taillights)
    - Located in the lower 60% of the frame  
    - Wider than they are tall (vehicle bodies) or very small bright spots (headlights)
    
    Real traffic signs are mounted on poles ABOVE the road surface, so they appear
    in the upper/middle portion of the frame.
    """
    center_y = y + h / 2.0
    y_ratio = center_y / max(1, img_h)
    area = w * h

    # Large red blob in the lower 55% of the frame is likely a vehicle
    if y_ratio > 0.55 and area > 400:
        return True

    # Medium-sized blob very low in the frame (bottom 35%)
    if y_ratio > 0.65 and area > 150:
        return True

    # Very large blob anywhere below the midpoint
    if y_ratio > 0.45 and area > 2000:
        return True

    return False


def _is_traffic_light(image_gray: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    """Detect and reject traffic lights based on tall-narrow aspect and internal bright spots.

    Traffic lights are distinctly taller than wide and contain multiple
    vertically stacked circular light spots.
    """
    aspect = float(w) / max(1, h)

    # Traffic lights are taller than wide with aspect < 0.55
    if aspect < 0.50 and h > 20:
        crop = image_gray[y:y + h, x:x + w]
        if crop.size == 0:
            return False
        _, bright = cv2.threshold(crop, 160, 255, cv2.THRESH_BINARY)
        contours_bright, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_bright) >= 2:
            return True
    return False


def _is_large_background_region(
    image_gray: np.ndarray,
    x: int, y: int, w: int, h: int,
) -> bool:
    """Reject large regions that are likely buildings, sky, or road surfaces.

    These regions have either very low variance (solid surfaces) or very high
    variance with no clear structure (busy textures like foliage).
    """
    area = w * h
    if area < 800:
        return False

    crop = image_gray[y:y + h, x:x + w]
    if crop.size == 0:
        return False

    variance = float(np.var(crop))

    # Very large regions are almost certainly not traffic signs
    if area > 5000:
        return True

    # Large regions with extreme variance (noisy textures like foliage)
    if area > 1500 and variance > 2000:
        return True

    # Large-ish regions with very low variance (solid walls, sky)
    if area > 800 and variance < 150:
        return True

    return False


def extract_candidate_regions(
    binary_mask: np.ndarray,
    config: Stage2Config,
    aux_masks: Optional[Dict[str, np.ndarray]] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    enhanced_bgr: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray], np.ndarray]:
    """Extract candidate bounding boxes with rigorous multi-criteria false positive rejection.

    Args:
        binary_mask: Cleaned binary mask from morphological processing.
        config: Stage2Config with all threshold parameters.
        aux_masks: Dict containing 'mask_white', 'mask_yellow', 'mask_red', 'mask_blue',
                   and 'mask_combined'.
        image_shape: (height, width) of the parent image.
        enhanced_bgr: Optional enhanced BGR image for edge analysis.

    Returns:
        (bounding_boxes, contours, verified_mask): Validated traffic sign candidates.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes: List[Tuple[int, int, int, int]] = []
    filtered_contours: List[np.ndarray] = []

    img_h = image_shape[0] if image_shape else binary_mask.shape[0]
    img_w = image_shape[1] if image_shape else binary_mask.shape[1]

    white_mask = aux_masks.get("mask_white") if aux_masks else None
    yellow_mask = aux_masks.get("mask_yellow") if aux_masks else None
    red_mask = aux_masks.get("mask_red") if aux_masks else None
    blue_mask = aux_masks.get("mask_blue") if aux_masks else None
    combined_mask = aux_masks.get("mask_combined") if aux_masks else None

    # Prepare grayscale for edge and texture analysis
    if enhanced_bgr is not None:
        image_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = None

    is_cropped_patch = (img_h < 150 and img_w < 150)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # ---- Filter 1: Area bounds ----
        if area < config.min_area or area > config.max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # ---- Filter 2: Bounding box dimension bounds ----
        if w < config.min_box_dim or h < config.min_box_dim:
            continue
        if hasattr(config, 'max_box_dim') and (w > config.max_box_dim or h > config.max_box_dim):
            continue

        # ---- Filter 3: Aspect ratio ----
        aspect_ratio = float(w) / max(1, h)
        if not (config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio):
            continue

        # ---- Filter 4: Spatial position ----
        if not is_cropped_patch:
            if y > int(config.max_road_y_ratio * img_h):
                continue
            if hasattr(config, 'min_sky_y_ratio') and (y + h) < int(config.min_sky_y_ratio * img_h):
                continue

        # ---- Filter 5: Circularity ----
        circularity = _compute_circularity(cnt)
        if hasattr(config, 'min_circularity') and circularity < config.min_circularity:
            continue

        # ---- Filter 6: Convexity ----
        convexity = _compute_convexity(cnt)
        if hasattr(config, 'min_convexity') and convexity < config.min_convexity:
            continue

        # ---- Filter 7: Color density ----
        if combined_mask is not None:
            crop_combined = combined_mask[y:y + h, x:x + w]
            color_density = _compute_color_density(crop_combined, w, h)
            if hasattr(config, 'min_color_density') and color_density < config.min_color_density:
                continue
            if hasattr(config, 'max_color_density') and color_density > config.max_color_density:
                continue

        # ---- Filter 8: Edge density ----
        if image_gray is not None:
            edge_density = _compute_edge_density(image_gray, x, y, w, h)
            if hasattr(config, 'min_edge_density') and edge_density < config.min_edge_density:
                continue
            if hasattr(config, 'max_edge_density') and edge_density > config.max_edge_density:
                continue

        # ---- Filter 9: Vehicle/taillight rejection ----
        if not is_cropped_patch:
            if _is_likely_vehicle_region(x, y, w, h, img_h, img_w, red_mask):
                continue

        # ---- Filter 10: Traffic light rejection ----
        if image_gray is not None and not is_cropped_patch:
            if _is_traffic_light(image_gray, x, y, w, h):
                continue

        # ---- Filter 11: Large background region rejection ----
        if image_gray is not None and not is_cropped_patch:
            if _is_large_background_region(image_gray, x, y, w, h):
                continue

        # ---- Filter 12: Structural verification ----
        if config.require_sign_structure and not is_cropped_patch:
            crop_w = white_mask[y:y + h, x:x + w] if white_mask is not None else None
            crop_y = yellow_mask[y:y + h, x:x + w] if yellow_mask is not None else None
            crop_r = red_mask[y:y + h, x:x + w] if red_mask is not None else None
            crop_b = blue_mask[y:y + h, x:x + w] if blue_mask is not None else None

            w_count = cv2.countNonZero(crop_w) if crop_w is not None else 0
            y_count = cv2.countNonZero(crop_y) if crop_y is not None else 0
            r_count = cv2.countNonZero(crop_r) if crop_r is not None else 0
            b_count = cv2.countNonZero(crop_b) if crop_b is not None else 0

            total_box_pixels = max(1, w * h)
            sign_color_frac = float(r_count + b_count + y_count) / total_box_pixels

            # A valid sign has:
            # (a) Sign color covering a reasonable fraction of the bbox, OR
            # (b) White interior content combined with sign border color, OR
            # (c) High convexity + circularity indicating a well-formed shape, OR
            # (d) Yellow content (warning signs)
            has_color_content = sign_color_frac >= 0.08
            has_white_interior = w_count >= 5 and (r_count >= 3 or b_count >= 3)
            has_solid_shape = convexity >= 0.60 and circularity >= 0.30
            has_yellow_content = y_count >= 5

            if not (has_color_content or has_white_interior or has_solid_shape or has_yellow_content):
                continue

        bounding_boxes.append((x, y, w, h))
        filtered_contours.append(cnt)

    # Generate verified binary mask containing ONLY confirmed traffic signs
    verified_mask = np.zeros_like(binary_mask)
    if filtered_contours:
        cv2.drawContours(verified_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

    return bounding_boxes, filtered_contours, verified_mask


def draw_segmentation_overlay(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
    bounding_boxes: List[Tuple[int, int, int, int]],
    color_mask: Tuple[int, int, int] = (0, 255, 0),
    box_color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.35
) -> np.ndarray:
    """Create a visual overlay of the segmented mask and detected candidate boxes on the image."""
    overlay = image_bgr.copy()

    # Highlight segmented mask pixels
    mask_bool = binary_mask > 0
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * np.array(color_mask, dtype=np.uint8)
    ).astype(np.uint8)

    # Draw bounding boxes and labels
    for idx, (x, y, w, h) in enumerate(bounding_boxes):
        cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, 2)
        label = f"Sign #{idx + 1}"
        cv2.putText(
            overlay,
            label,
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA
        )

    return overlay
