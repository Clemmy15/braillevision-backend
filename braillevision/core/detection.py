"""Braille dot detection using blob and contour analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectedDot:
    """A single detected Braille dot."""

    x: int
    y: int
    radius: float
    confidence: float
    area: float
    circularity: float = 1.0
    sources: int = 1

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass
class DetectionDebug:
    accepted: List[DetectedDot] = field(default_factory=list)
    rejected: List[DetectedDot] = field(default_factory=list)
    masks: Dict[str, np.ndarray] = field(default_factory=dict)


class DotDetector:
    """Detect raised Braille dots from preprocessed images."""

    def __init__(
        self,
        min_area_ratio: float = 0.0008,
        max_area_ratio: float = 0.015,
        min_circularity: float = 0.52,
        min_convexity: float = 0.72,
        max_aspect_ratio: float = 1.65,
        merge_distance_ratio: float = 0.6,
        dilate_iterations: int = 1,
        dilate_kernel: int = 5,
        edge_margin: int = 22,
        radius_tolerance: float = 1.55,
        min_dot_confidence: float = 38.0,
        min_neighbors: int = 1,
        require_consensus: bool = True,
        emboss_min_circularity: float = 0.32,
        emboss_min_convexity: float = 0.48,
        emboss_max_aspect: float = 2.4,
    ) -> None:
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_circularity = min_circularity
        self.min_convexity = min_convexity
        self.max_aspect_ratio = max_aspect_ratio
        self.merge_distance_ratio = merge_distance_ratio
        self.dilate_iterations = dilate_iterations
        self.dilate_kernel = dilate_kernel if dilate_kernel % 2 == 1 else dilate_kernel + 1
        self.edge_margin = edge_margin
        self.radius_tolerance = radius_tolerance
        self.min_dot_confidence = min_dot_confidence
        self.min_neighbors = min_neighbors
        self.require_consensus = require_consensus
        self.emboss_min_circularity = emboss_min_circularity
        self.emboss_min_convexity = emboss_min_convexity
        self.emboss_max_aspect = emboss_max_aspect
        self.last_debug = DetectionDebug()

    def _area_bounds(self, image_shape: Tuple[int, ...]) -> Tuple[float, float]:
        height, width = image_shape[:2]
        image_area = float(height * width)
        return self.min_area_ratio * image_area, self.max_area_ratio * image_area

    def _adaptive_area_bounds(
        self,
        image_shape: Tuple[int, ...],
        grayscale: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Widen area limits when Otsu finds large filled circles (print/screenshot Braille)."""
        min_area, max_area = self._area_bounds(image_shape)
        if grayscale is None:
            return min_area, max_area

        height, width = image_shape[:2]
        image_area = float(height * width)
        _, mask = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blob_areas: List[float] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < image_area * 0.002:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity >= 0.45:
                blob_areas.append(area)

        if not blob_areas:
            return min_area, max_area

        blob_areas = [area for area in blob_areas if area <= image_area * 0.08]
        if not blob_areas:
            return min_area, max_area

        median_area = float(np.median(blob_areas))
        min_area = max(image_area * 0.0003, median_area * 0.12)
        max_area = min(image_area * 0.22, median_area * 5.0)
        max_area = max(max_area, min_area * 3.0)
        return min_area, max_area

    def _is_high_contrast_print(self, grayscale: np.ndarray) -> bool:
        """True for solid black dots on white (screenshots, print, digital cards)."""
        _, mask = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        image_area = float(grayscale.shape[0] * grayscale.shape[1])
        circular = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_ratio = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            largest_ratio = max(largest_ratio, area / image_area)
            if area < image_area * 0.004 or area > image_area * 0.12:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            if 4.0 * np.pi * area / (perimeter * perimeter) >= 0.6:
                circular += 1
        if largest_ratio > 0.2:
            return False
        return circular >= 2

    def _detect_printed_dots(
        self,
        grayscale: np.ndarray,
        min_area: float,
        max_area: float,
    ) -> List[DetectedDot]:
        """Detect filled circular dots via global Otsu (not emboss)."""
        _, mask = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return self._detect_with_contours(mask, min_area, max_area)

    def _build_blob_detector(self, min_area: float, max_area: float) -> cv2.SimpleBlobDetector:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = max(min_area, 20.0)
        params.maxArea = max_area
        params.filterByCircularity = True
        params.minCircularity = self.min_circularity
        params.filterByConvexity = True
        params.minConvexity = self.min_convexity
        params.filterByInertia = True
        params.minInertiaRatio = 0.35
        params.filterByColor = False
        return cv2.SimpleBlobDetector_create(params)

    def _emboss_kernel_sizes(self, enhanced: np.ndarray) -> List[int]:
        short_side = min(enhanced.shape[:2])
        if short_side < 220:
            return [3, 5, 7, 9]
        if short_side < 450:
            return [5, 7, 9, 11]
        return [7, 9, 11, 15]

    def _build_emboss_mask(self, enhanced: np.ndarray, kernel_size: Optional[int] = None) -> np.ndarray:
        if kernel_size is None:
            kernel_size = max(int(min(enhanced.shape[:2]) * 0.028), 7)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
        combined = cv2.addWeighted(tophat, 0.6, blackhat, 0.4, 0)
        combined = cv2.GaussianBlur(combined, (3, 3), 0)
        _, mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        return mask

    def _detect_emboss_multiscale(
        self,
        enhanced: np.ndarray,
        min_area: float,
        max_area: float,
    ) -> List[DetectedDot]:
        """Multi-scale top-hat/black-hat for soft/low-contrast embossed dots."""
        emboss_min = max(22.0, min_area * 0.2)
        emboss_max = max(max_area * 2.5, min_area * 12.0, 3500.0)
        source_lists: List[List[DetectedDot]] = []
        for kernel_size in self._emboss_kernel_sizes(enhanced):
            mask = self._build_emboss_mask(enhanced, kernel_size=kernel_size)
            contours = self._detect_emboss_contours(mask, emboss_min, emboss_max)
            blobs = self._detect_emboss_blobs(mask, emboss_min, emboss_max)
            picks = contours if len(contours) >= len(blobs) else blobs
            if picks:
                source_lists.append(picks)
        if not source_lists:
            return []
        return self._merge_with_sources(source_lists)

    def _detect_emboss_contours(
        self,
        mask: np.ndarray,
        min_area: float,
        max_area: float,
    ) -> List[DetectedDot]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dots: List[DetectedDot] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            circularity, convexity, aspect = self._contour_shape_metrics(contour)
            if circularity < self.emboss_min_circularity:
                continue
            if convexity < self.emboss_min_convexity:
                continue
            if aspect > self.emboss_max_aspect:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            confidence = self._score_dot(area, radius, min_area, max_area, circularity, convexity)
            dots.append(
                DetectedDot(
                    x=int(x),
                    y=int(y),
                    radius=float(radius),
                    confidence=confidence,
                    area=float(area),
                    circularity=float(circularity),
                )
            )
        return dots

    def _detect_emboss_blobs(
        self,
        mask: np.ndarray,
        min_area: float,
        max_area: float,
    ) -> List[DetectedDot]:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = max(min_area, 15.0)
        params.maxArea = max_area
        params.filterByCircularity = True
        params.minCircularity = self.emboss_min_circularity
        params.filterByConvexity = True
        params.minConvexity = self.emboss_min_convexity
        params.filterByInertia = True
        params.minInertiaRatio = 0.2
        params.filterByColor = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(mask)
        dots: List[DetectedDot] = []
        for kp in keypoints:
            radius = float(kp.size / 2.0)
            area = float(np.pi * radius * radius)
            confidence = self._score_dot(area, radius, min_area, max_area, 0.9, 0.9)
            dots.append(
                DetectedDot(
                    x=int(kp.pt[0]),
                    y=int(kp.pt[1]),
                    radius=radius,
                    confidence=confidence,
                    area=area,
                    circularity=0.9,
                )
            )
        return dots

    def _local_contrast_mask(self, enhanced: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(enhanced, (21, 21), 0)
        diff = cv2.absdiff(enhanced, blur)
        diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    def _prepare_detection_mask(self, binary: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.dilate_kernel, self.dilate_kernel)
        )
        return cv2.dilate(binary, kernel, iterations=self.dilate_iterations)

    def _detect_with_blobs(
        self, mask: np.ndarray, min_area: float, max_area: float
    ) -> List[DetectedDot]:
        detector = self._build_blob_detector(min_area, max_area)
        keypoints = detector.detect(mask)
        dots: List[DetectedDot] = []

        for kp in keypoints:
            radius = float(kp.size / 2.0)
            area = np.pi * radius * radius
            confidence = self._score_dot(area, radius, min_area, max_area, 1.0, 1.0)
            dots.append(
                DetectedDot(
                    x=int(kp.pt[0]),
                    y=int(kp.pt[1]),
                    radius=radius,
                    confidence=confidence,
                    area=area,
                    circularity=1.0,
                )
            )
        return dots

    def _contour_shape_metrics(self, contour: np.ndarray) -> Tuple[float, float, float]:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0.0
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        aspect = max(width, height) / max(min(width, height), 1.0)
        return circularity, convexity, aspect

    def _detect_with_contours(
        self, mask: np.ndarray, min_area: float, max_area: float
    ) -> List[DetectedDot]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dots: List[DetectedDot] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            circularity, convexity, aspect = self._contour_shape_metrics(contour)
            if circularity < self.min_circularity:
                continue
            if convexity < self.min_convexity:
                continue
            if aspect > self.max_aspect_ratio:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            confidence = self._score_dot(area, radius, min_area, max_area, circularity, convexity)
            dots.append(
                DetectedDot(
                    x=int(x),
                    y=int(y),
                    radius=float(radius),
                    confidence=confidence,
                    area=float(area),
                    circularity=float(circularity),
                )
            )
        return dots

    def _detect_from_mask(
        self, mask: np.ndarray, min_area: float, max_area: float, prefer_blobs: bool
    ) -> List[DetectedDot]:
        prepared = self._prepare_detection_mask(mask)
        dots: List[DetectedDot] = []
        if prefer_blobs:
            dots = self._detect_with_blobs(prepared, min_area, max_area)
        contour_dots = self._detect_with_contours(prepared, min_area, max_area)
        if len(contour_dots) > len(dots):
            dots = contour_dots
        return dots

    def _score_dot(
        self,
        area: float,
        radius: float,
        min_area: float,
        max_area: float,
        circularity: float = 1.0,
        convexity: float = 1.0,
    ) -> float:
        mid_area = (min_area + max_area) / 2.0
        area_score = 1.0 - min(abs(area - mid_area) / mid_area, 1.0)
        size_score = min(radius / 10.0, 1.0)
        shape_score = min(circularity * 0.65 + convexity * 0.35, 1.0)
        return round((0.5 * area_score + 0.25 * size_score + 0.25 * shape_score) * 100, 1)

    def _merge_with_sources(self, dot_lists: List[List[DetectedDot]]) -> List[DetectedDot]:
        tagged: List[Tuple[DetectedDot, int]] = []
        for source_id, dots in enumerate(dot_lists):
            for dot in dots:
                tagged.append((dot, source_id))
        if not tagged:
            return []

        avg_radius = float(np.mean([dot.radius for dot, _ in tagged]))
        merge_distance = max(avg_radius * self.merge_distance_ratio, 8.0)
        merged: List[DetectedDot] = []
        used = [False] * len(tagged)

        for index, (dot, source_id) in enumerate(tagged):
            if used[index]:
                continue
            cluster = [dot]
            source_ids = {source_id}
            used[index] = True
            for other_index in range(index + 1, len(tagged)):
                if used[other_index]:
                    continue
                other, other_source = tagged[other_index]
                if np.hypot(dot.x - other.x, dot.y - other.y) <= merge_distance:
                    cluster.append(other)
                    source_ids.add(other_source)
                    used[other_index] = True

            weights = [item.confidence for item in cluster]
            total = sum(weights) or 1.0
            x = int(sum(item.x * weight for item, weight in zip(cluster, weights)) / total)
            y = int(sum(item.y * weight for item, weight in zip(cluster, weights)) / total)
            radius = float(np.mean([item.radius for item in cluster]))
            area = float(np.mean([item.area for item in cluster]))
            confidence = float(np.mean([item.confidence for item in cluster]))
            circularity = float(np.mean([item.circularity for item in cluster]))
            merged.append(
                DetectedDot(
                    x=x,
                    y=y,
                    radius=radius,
                    confidence=confidence,
                    area=area,
                    circularity=circularity,
                    sources=len(source_ids),
                )
            )
        return merged

    def _filter_consensus(self, dots: List[DetectedDot]) -> Tuple[List[DetectedDot], List[DetectedDot]]:
        if not self.require_consensus:
            return dots, []
        if len(dots) >= 10:
            single_source = sum(1 for dot in dots if dot.sources < 2)
            if single_source / len(dots) >= 0.65:
                return dots, []
        kept: List[DetectedDot] = []
        rejected: List[DetectedDot] = []
        for dot in dots:
            passes = (
                dot.sources >= 2
                or dot.confidence >= 50.0
                or (dot.circularity >= 0.5 and dot.confidence >= 38.0)
            )
            if passes:
                kept.append(dot)
            else:
                rejected.append(dot)
        if len(kept) < max(8, len(dots) // 4):
            return dots, rejected
        return kept, rejected

    def _filter_neighborhood(self, dots: List[DetectedDot]) -> Tuple[List[DetectedDot], List[DetectedDot]]:
        if len(dots) < 10:
            return dots, []

        radii = np.array([dot.radius for dot in dots], dtype=np.float32)
        median_radius = float(np.median(radii))
        neighbor_radius = max(median_radius * 6.5, 35.0)
        coords = np.array([[dot.x, dot.y] for dot in dots], dtype=np.float32)

        kept: List[DetectedDot] = []
        rejected: List[DetectedDot] = []
        for index, dot in enumerate(dots):
            others = np.delete(coords, index, axis=0)
            distances = np.linalg.norm(others - coords[index], axis=1)
            neighbors = int(np.sum(distances <= neighbor_radius))
            if neighbors >= self.min_neighbors or dot.confidence >= 58.0:
                kept.append(dot)
            else:
                rejected.append(dot)
        return kept if len(kept) >= 3 else dots, rejected

    def _filter_isolated_noise(self, dots: List[DetectedDot]) -> List[DetectedDot]:
        if len(dots) < 8:
            return dots

        coords = np.array([[dot.x, dot.y] for dot in dots], dtype=np.float32)
        kept: List[DetectedDot] = []
        for index, dot in enumerate(dots):
            others = np.delete(coords, index, axis=0)
            distances = np.linalg.norm(others - coords[index], axis=1)
            nearest = float(np.min(distances)) if distances.size else 0.0
            max_gap = max(dot.radius * 6.0, 48.0)
            if nearest <= max_gap or dot.confidence >= 42.0:
                kept.append(dot)
        return kept if len(kept) >= max(6, len(dots) // 3) else dots

    def _filter_low_confidence(self, dots: List[DetectedDot]) -> List[DetectedDot]:
        threshold = self.min_dot_confidence
        if len(dots) >= 12:
            threshold = max(26.0, self.min_dot_confidence - 14.0)
        filtered = [dot for dot in dots if dot.confidence >= threshold]
        return filtered if len(filtered) >= max(6, len(dots) // 4) else dots

    def _filter_edge_dots(self, dots: List[DetectedDot], shape: Tuple[int, ...]) -> List[DetectedDot]:
        if not dots:
            return dots

        height, width = shape[:2]
        margin = max(6, int(min(height, width) * 0.035), self.edge_margin // 2)
        inside = [
            dot
            for dot in dots
            if margin <= dot.x <= width - margin and margin <= dot.y <= height - margin
        ]
        if len(inside) >= max(2, len(dots) - 1):
            return inside

        soft = max(4, margin // 2)
        relaxed: List[DetectedDot] = []
        for dot in dots:
            in_frame = soft <= dot.x <= width - soft and soft <= dot.y <= height - soft
            if in_frame or dot.confidence >= 52.0 or dot.circularity >= 0.5:
                relaxed.append(dot)
        return relaxed if len(relaxed) >= 2 else dots

    def _filter_by_radius(self, dots: List[DetectedDot]) -> List[DetectedDot]:
        if len(dots) < 4:
            return dots

        radii = np.array([dot.radius for dot in dots], dtype=np.float32)
        median_radius = float(np.median(radii))
        tolerance = self.radius_tolerance
        if len(dots) >= 12:
            tolerance = max(tolerance, 1.95)
        lower = median_radius / tolerance
        upper = median_radius * tolerance
        filtered = [dot for dot in dots if lower <= dot.radius <= upper]
        return filtered if len(filtered) >= max(6, len(dots) // 3) else dots

    def detect(
        self,
        binary: np.ndarray,
        grayscale: Optional[np.ndarray] = None,
        enhanced: Optional[np.ndarray] = None,
        prefer_blobs: bool = True,
    ) -> List[DetectedDot]:
        min_area, max_area = self._area_bounds(binary.shape)
        print_style = False
        if grayscale is not None:
            min_area, max_area = self._adaptive_area_bounds(binary.shape, grayscale)
            print_style = self._is_high_contrast_print(grayscale)

        source_lists: List[List[DetectedDot]] = []
        masks: Dict[str, np.ndarray] = {}

        if grayscale is not None and print_style:
            printed = self._detect_printed_dots(grayscale, min_area, max_area)
            if len(printed) >= 2:
                _, printed_mask = cv2.threshold(
                    grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                masks["printed"] = printed_mask
                source_lists.append(printed)
            else:
                print_style = False

        if enhanced is not None and not print_style:
            emboss_dots = self._detect_emboss_multiscale(enhanced, min_area, max_area)
            if emboss_dots:
                masks["emboss"] = self._build_emboss_mask(enhanced)
                source_lists.append(emboss_dots)

            local_mask = self._local_contrast_mask(enhanced)
            masks["local_contrast"] = local_mask
            local_dots = self._detect_emboss_contours(local_mask, min_area, max_area)
            if local_dots:
                source_lists.append(local_dots)

        masks["binary"] = binary
        source_lists.append(self._detect_from_mask(binary, min_area, max_area, prefer_blobs))

        if print_style and grayscale is not None:
            _, otsu_mask = cv2.threshold(
                grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            source_lists.append(
                self._detect_from_mask(otsu_mask, min_area, max_area, prefer_blobs=False)
            )

        if len(sum(source_lists, [])) < 3 and grayscale is not None:
            fallback = cv2.adaptiveThreshold(
                grayscale,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                15,
                4,
            )
            source_lists.append(self._detect_from_mask(fallback, min_area, max_area, prefer_blobs))

        all_candidates = self._merge_with_sources(source_lists)
        all_candidates.sort(key=lambda item: (item.y, item.x))

        rejected: List[DetectedDot] = []
        dots = self._filter_edge_dots(all_candidates, binary.shape)
        dots = self._filter_by_radius(dots)
        dots, rejected_consensus = self._filter_consensus(dots)
        rejected.extend(rejected_consensus)
        dots, rejected_neighbors = self._filter_neighborhood(dots)
        rejected.extend(rejected_neighbors)
        dots = self._filter_low_confidence(dots)
        dots = self._filter_isolated_noise(dots)

        self.last_debug = DetectionDebug(accepted=dots, rejected=rejected, masks=masks)
        return dots

    def build_emboss_mask(self, enhanced: np.ndarray) -> np.ndarray:
        return self._build_emboss_mask(enhanced)

    def overall_confidence(self, dots: List[DetectedDot]) -> float:
        if not dots:
            return 0.0
        scores = []
        for dot in dots:
            consensus_bonus = min(dot.sources - 1, 2) * 4.0
            shape_bonus = min(dot.circularity, 1.0) * 8.0
            scores.append(min(dot.confidence + consensus_bonus + shape_bonus, 100.0))
        return round(float(np.mean(scores)), 1)
