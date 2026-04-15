from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


COLOR_KEYS = ("W", "R", "G", "Y", "O", "B")

# HSV prototypes used for fallback nearest-color classification.
# OpenCV HSV: H in [0,179], S/V in [0,255].
HSV_PROTOTYPES: dict[str, tuple[int, int, int]] = {
    "W": (0, 20, 230),
    "R": (0, 180, 180),
    "O": (17, 200, 200),
    "Y": (30, 200, 220),
    "G": (60, 190, 180),
    "B": (110, 200, 180),
}


@dataclass
class DetectionResult:
    grid: list[str]  # 9 letters in row-major order
    confidence: list[float]
    roi: dict[str, int]


def _gray_world_white_balance(bgr: np.ndarray) -> np.ndarray:
    """
    Simple gray-world white balance to reduce warm/cool lighting bias.
    """
    bgr_f = bgr.astype(np.float32)
    b_mean = float(np.mean(bgr_f[:, :, 0]))
    g_mean = float(np.mean(bgr_f[:, :, 1]))
    r_mean = float(np.mean(bgr_f[:, :, 2]))
    avg = (b_mean + g_mean + r_mean) / 3.0

    if b_mean > 0:
        bgr_f[:, :, 0] *= avg / b_mean
    if g_mean > 0:
        bgr_f[:, :, 1] *= avg / g_mean
    if r_mean > 0:
        bgr_f[:, :, 2] *= avg / r_mean

    return np.clip(bgr_f, 0, 255).astype(np.uint8)


def _circular_hue_mean(h_values: np.ndarray) -> int:
    """
    Circular mean for OpenCV hue values in [0, 179].
    """
    if h_values.size == 0:
        return 0
    radians = (h_values.astype(np.float32) * 2.0) * (np.pi / 180.0)
    sin_m = float(np.mean(np.sin(radians)))
    cos_m = float(np.mean(np.cos(radians)))
    angle = np.arctan2(sin_m, cos_m)
    if angle < 0:
        angle += 2 * np.pi
    hue_deg = angle * (180.0 / np.pi)
    return int((hue_deg / 2.0) % 180)


def _robust_hsv_from_patch(patch_hsv: np.ndarray) -> tuple[int, int, int]:
    pixels = patch_hsv.reshape(-1, 3)
    s = pixels[:, 1]
    v = pixels[:, 2]

    # Keep colorful and visible pixels to avoid black borders/shadows.
    colorful = pixels[(s > 45) & (v > 45)]
    if colorful.shape[0] >= 20:
        h_val = _circular_hue_mean(colorful[:, 0])
        s_val = int(np.percentile(colorful[:, 1], 60))
        v_val = int(np.percentile(colorful[:, 2], 60))
        return h_val, s_val, v_val

    # Fallback for white / low-saturation cells.
    h_val = _circular_hue_mean(pixels[:, 0])
    s_val = int(np.percentile(pixels[:, 1], 50))
    v_val = int(np.percentile(pixels[:, 2], 65))
    return h_val, s_val, v_val



def _classify_hsv(h: int, s: int, v: int) -> tuple[str, float]:
    # Rule-based fast path.
    if s < 50 and v > 115:
        return "W", 0.95

    if (h < 9 or h >= 170) and s > 60 and v > 45:
        return "R", 0.92
    if 9 <= h < 24 and s > 75 and v > 50:
        return "O", 0.90
    if 24 <= h < 40 and s > 70 and v > 55:
        return "Y", 0.90
    if 40 <= h < 90 and s > 60 and v > 45:
        return "G", 0.88
    if 90 <= h < 140 and s > 60 and v > 45:
        return "B", 0.88

    # Fallback nearest prototype in HSV space (cyclic hue distance).
    best_color = "W"
    best_dist = float("inf")
    for color, (ph, ps, pv) in HSV_PROTOTYPES.items():
        hue_dist = min(abs(h - ph), 180 - abs(h - ph))
        dist = (hue_dist * 2.0) ** 2 + ((s - ps) * 0.7) ** 2 + ((v - pv) * 0.6) ** 2
        if dist < best_dist:
            best_dist = dist
            best_color = color

    confidence = float(max(0.40, min(0.85, 1.0 - (best_dist / 60000.0))))
    return best_color, confidence



def _center_square_roi(image: np.ndarray, ratio: float = 0.58) -> tuple[np.ndarray, dict[str, int]]:
    h, w = image.shape[:2]
    side = int(min(h, w) * ratio)
    x = (w - side) // 2
    y = (h - side) // 2
    roi = image[y : y + side, x : x + side]
    return roi, {"x": x, "y": y, "size": side}



def detect_face_from_image_bytes(image_bytes: bytes) -> DetectionResult:
    np_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")

    bgr = _gray_world_white_balance(bgr)

    roi_bgr, roi_meta = _center_square_roi(bgr)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    side = hsv.shape[0]
    cell = side // 3

    grid: list[str] = []
    confidence: list[float] = []

    for r in range(3):
        for c in range(3):
            y0 = r * cell
            x0 = c * cell
            y1 = side if r == 2 else (r + 1) * cell
            x1 = side if c == 2 else (c + 1) * cell

            patch = hsv[y0:y1, x0:x1]
            if patch.size == 0:
                raise ValueError("Empty cell patch detected")

            # Sample central region for stability.
            ph, pw = patch.shape[:2]
            cy0, cy1 = int(ph * 0.30), int(ph * 0.70)
            cx0, cx1 = int(pw * 0.30), int(pw * 0.70)
            core = patch[cy0:cy1, cx0:cx1]
            if core.size == 0:
                core = patch

            h, s, v = _robust_hsv_from_patch(core)

            color, conf = _classify_hsv(h, s, v)
            grid.append(color)
            confidence.append(round(conf, 3))

    return DetectionResult(grid=grid, confidence=confidence, roi=roi_meta)



def make_color_grid_payload(grid: list[str]) -> list[list[str]]:
    if len(grid) != 9:
        raise ValueError("Face grid must contain 9 color values")
    return [grid[0:3], grid[3:6], grid[6:9]]



def as_debug_dict(result: DetectionResult) -> dict[str, Any]:
    return {
        "grid": result.grid,
        "grid_2d": make_color_grid_payload(result.grid),
        "confidence": result.confidence,
        "roi": result.roi,
    }
