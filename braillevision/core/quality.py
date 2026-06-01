"""Camera capture quality analysis for BrailleVision."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class QualityStatus(str, Enum):
    GOOD = "good"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    TOO_BLURRY = "too_blurry"
    MOVE_CLOSER = "move_closer"


@dataclass
class QualityReport:
    """Image quality metrics and user-facing feedback."""

    blur_score: float
    brightness: float
    status: QualityStatus
    message: str
    is_acceptable: bool

    def __str__(self) -> str:
        return (
            f"Quality: {self.message} | blur={self.blur_score:.1f}, "
            f"brightness={self.brightness:.1f}"
        )


class QualityAnalyzer:
    """Evaluate whether an image is suitable for Braille detection."""

    def __init__(
        self,
        blur_threshold: float = 80.0,
        dark_threshold: float = 60.0,
        bright_threshold: float = 240.0,
        min_dot_area_ratio: float = 0.00008,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.min_dot_area_ratio = min_dot_area_ratio

    def measure_blur(self, gray: np.ndarray) -> float:
        """
        Laplacian variance, scaled to a reference resolution.

        Small/low-res images look artificially 'blurry' without scaling; embossed
        Braille with soft shadows also scores lower than sharp print.
        """
        height, width = gray.shape[:2]
        image_area = float(height * width)
        reference_area = 480.0 * 640.0
        raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if image_area < reference_area:
            scaled = raw * (reference_area / max(image_area, 1.0))
        else:
            scaled = raw
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_raw = float(cv2.Laplacian(enhanced, cv2.CV_64F).var())
        if image_area < reference_area:
            enhanced_scaled = enhanced_raw * (reference_area / max(image_area, 1.0))
        else:
            enhanced_scaled = enhanced_raw
        return max(scaled, enhanced_scaled * 0.85)

    def measure_brightness(self, gray: np.ndarray) -> float:
        return float(np.mean(gray))

    def analyze(
        self,
        gray: np.ndarray,
        binary: np.ndarray | None = None,
        dots_detected: int = 0,
    ) -> QualityReport:
        blur_score = self.measure_blur(gray)
        brightness = self.measure_brightness(gray)

        status = QualityStatus.GOOD
        message = "Good capture"

        if brightness < self.dark_threshold:
            status = QualityStatus.TOO_DARK
            message = "Too dark — increase lighting"
        elif brightness > self.bright_threshold:
            status = QualityStatus.TOO_BRIGHT
            message = "Too bright — reduce glare or lighting"
        elif blur_score < self.blur_threshold and dots_detected < 12:
            status = QualityStatus.TOO_BLURRY
            message = "Low sharpness — move closer or improve lighting"
        elif binary is not None:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image_area = float(gray.shape[0] * gray.shape[1])
            min_area = self.min_dot_area_ratio * image_area
            valid = [c for c in contours if cv2.contourArea(c) >= min_area]
            if len(valid) < 3:
                status = QualityStatus.MOVE_CLOSER
                message = "Move camera closer — dots too small"

        is_acceptable = status == QualityStatus.GOOD
        return QualityReport(
            blur_score=blur_score,
            brightness=brightness,
            status=status,
            message=message,
            is_acceptable=is_acceptable,
        )
