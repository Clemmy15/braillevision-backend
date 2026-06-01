"""Image preprocessing for physical Braille detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .alignment import auto_perspective_correct, deskew_image, estimate_skew_angle


@dataclass
class PreprocessResult:
    """Outputs from the preprocessing pipeline."""

    original: np.ndarray
    grayscale: np.ndarray
    enhanced: np.ndarray
    blurred: np.ndarray
    binary: np.ndarray
    skew_angle: float = 0.0
    binary_candidates: List[np.ndarray] = field(default_factory=list)


class Preprocessor:
    """Prepare camera images for embossed Braille dot detection."""

    def __init__(
        self,
        blur_kernel: int = 5,
        clahe_clip: float = 2.5,
        clahe_grid: Tuple[int, int] = (8, 8),
        adaptive_block_size: int = 21,
        adaptive_c: int = 8,
        morph_kernel: int = 3,
        auto_align: bool = True,
    ) -> None:
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.adaptive_block_size = (
            adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1
        )
        self.adaptive_c = adaptive_c
        self.morph_kernel = morph_kernel
        self.auto_align = auto_align

    def load_image(self, path: str) -> np.ndarray:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return image

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image.copy()
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _dynamic_clahe(self, gray: np.ndarray) -> cv2.CLAHE:
        std = float(np.std(gray))
        clip = self.clahe_clip
        if std < 25.0:
            clip = min(4.0, clip + 1.2)
        elif std > 70.0:
            clip = max(1.8, clip - 0.6)
        return cv2.createCLAHE(clipLimit=clip, tileGridSize=self.clahe_grid)

    def normalize_local_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Reduce uneven lighting (common in phone photos)."""
        blur_sigma = max(int(min(gray.shape[:2]) * 0.04), 15)
        if blur_sigma % 2 == 0:
            blur_sigma += 1
        background = cv2.GaussianBlur(gray, (blur_sigma, blur_sigma), 0)
        normalized = cv2.divide(gray, background, scale=128)
        return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

    def enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        clahe = self._dynamic_clahe(gray)
        local = self.normalize_local_contrast(gray)
        enhanced = clahe.apply(local)
        shadow_boost = cv2.addWeighted(enhanced, 0.85, clahe.apply(gray), 0.15, 0)
        return shadow_boost

    def apply_blur(self, gray: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

    def adaptive_threshold(self, blurred: np.ndarray, c_value: Optional[int] = None) -> np.ndarray:
        c = self.adaptive_c if c_value is None else c_value
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adaptive_block_size,
            c,
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        return binary

    def _binary_candidates(self, blurred: np.ndarray) -> List[np.ndarray]:
        candidates: List[np.ndarray] = []
        for c in (6, 8, 10, 12):
            candidates.append(self.adaptive_threshold(blurred, c_value=c))
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        candidates.append(otsu)
        return candidates

    def _score_binary_mask(self, mask: np.ndarray) -> float:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        height, width = mask.shape[:2]
        image_area = float(height * width)
        min_area = image_area * 0.0008
        max_area = image_area * 0.015
        scores: List[float] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.45:
                continue
            scores.append(circularity)

        if not scores:
            return 0.0
        count_factor = min(len(scores) / 30.0, 1.0)
        return float(np.mean(scores)) * count_factor

    def select_best_binary(self, blurred: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        candidates = self._binary_candidates(blurred)
        best = candidates[0]
        best_score = -1.0
        for candidate in candidates:
            score = self._score_binary_mask(candidate)
            if score > best_score:
                best_score = score
                best = candidate
        return best, candidates

    def correct_perspective(
        self,
        image: np.ndarray,
        src_points: np.ndarray,
        dst_size: Tuple[int, int],
    ) -> np.ndarray:
        width, height = dst_size
        dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        matrix = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst)
        return cv2.warpPerspective(image, matrix, (width, height))

    def add_safe_border(self, image: np.ndarray, pad_ratio: float = 0.12) -> np.ndarray:
        """Add margin so dots near the frame edge are not clipped (common in phone photos)."""
        height, width = image.shape[:2]
        pad_y = max(int(height * pad_ratio), 12)
        pad_x = max(int(width * pad_ratio), 12)
        return cv2.copyMakeBorder(
            image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
        )

    def upscale_if_small(self, image: np.ndarray, min_height: int = 360) -> np.ndarray:
        height, width = image.shape[:2]
        if height >= min_height:
            return image
        scale = min_height / max(height, 1)
        return cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    def process(
        self,
        image: np.ndarray,
        perspective_points: Optional[np.ndarray] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> PreprocessResult:
        original = self.add_safe_border(image.copy())
        original = self.upscale_if_small(original)

        if self.auto_align:
            original = auto_perspective_correct(original)
            original = deskew_image(original)

        if perspective_points is not None and output_size is not None:
            original = self.correct_perspective(original, perspective_points, output_size)

        grayscale = self.to_grayscale(original)
        skew_angle = estimate_skew_angle(grayscale)
        enhanced = self.enhance_contrast(grayscale)
        blurred = self.apply_blur(enhanced)
        binary, candidates = self.select_best_binary(blurred)

        return PreprocessResult(
            original=original,
            grayscale=grayscale,
            enhanced=enhanced,
            blurred=blurred,
            binary=binary,
            skew_angle=skew_angle,
            binary_candidates=candidates,
        )
