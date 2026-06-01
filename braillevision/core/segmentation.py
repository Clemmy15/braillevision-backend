"""Group detected dots into Braille cells (2 columns x 3 rows)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .alignment import deskew_dots
from .braille_map import BRAILLE_MAP, hamming_distance, match_pattern
from .detection import DetectedDot


# Braille dot layout (2 columns x 3 rows):
#   1  4
#   2  5
#   3  6
# Index mapping: row + col * 3  (row 0..2, col 0..1)
GRID_LABELS = ("1", "2", "3", "4", "5", "6")


@dataclass
class BrailleCell:
    """A segmented Braille cell with dot occupancy pattern."""

    cell_id: int
    dots: List[DetectedDot]
    pattern: Tuple[bool, bool, bool, bool, bool, bool]
    confidence: float
    bounding_box: Tuple[int, int, int, int]
    grid_map: Dict[int, DetectedDot] = field(default_factory=dict)

    @property
    def x(self) -> int:
        return self.bounding_box[0]

    @property
    def y(self) -> int:
        return self.bounding_box[1]

    @property
    def width(self) -> int:
        return self.bounding_box[2]

    @property
    def height(self) -> int:
        return self.bounding_box[3]

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "dots": [int(value) for value in self.pattern],
            "confidence": round(self.confidence, 1),
            "bounding_box": self.bounding_box,
        }


@dataclass
class SegmentationResult:
    """Output of the cell segmentation pipeline."""

    cells: List[BrailleCell]
    confidence: float
    stability_score: float
    row_centers: Tuple[float, float, float]
    col_spacing: float
    row_spacing: float
    cell_spacing: float


@dataclass
class _SpacingMetrics:
    row_spacing: float
    col_spacing: float
    cell_spacing: float
    avg_radius: float


class CellSegmenter:
    """Spatial clustering of dots into standard Braille cells."""

    def __init__(
        self,
        min_dots_per_cell: int = 1,
        prefer_min_dots: int = 2,
        max_dots_per_cell: int = 6,
        row_count: int = 3,
        col_count: int = 2,
        row_merge_ratio: float = 0.45,
        col_merge_ratio: float = 0.55,
        cell_merge_ratio: float = 0.75,
        bbox_padding: float = 1.8,
    ) -> None:
        self.min_dots_per_cell = min_dots_per_cell
        self.prefer_min_dots = prefer_min_dots
        self.row_count = row_count
        self.col_count = col_count
        self.row_merge_ratio = row_merge_ratio
        self.col_merge_ratio = col_merge_ratio
        self.cell_merge_ratio = cell_merge_ratio
        self.bbox_padding = bbox_padding
        self.max_dots_per_cell = max_dots_per_cell

    def segment(
        self,
        dots: List[DetectedDot],
        skew_angle: float = 0.0,
    ) -> SegmentationResult:
        if not dots:
            return self._empty_result()

        if abs(skew_angle) >= 0.4:
            dots = deskew_dots(dots, skew_angle)

        row_centers, row_spacing, line_indices = self._fit_row_model(dots)
        line_dots = [dots[i] for i in line_indices]
        if len(line_dots) < self.min_dots_per_cell:
            line_dots = list(dots)
            row_centers, row_spacing, _ = self._fit_row_model(line_dots)

        metrics = self._estimate_spacing(line_dots, row_centers, row_spacing)
        row_assignments = self._assign_rows(line_dots, row_centers, metrics.row_spacing)
        cell_groups = self._group_cells(line_dots, row_assignments, metrics)
        cell_groups = self._merge_singleton_groups(line_dots, cell_groups, metrics)
        cells = self._build_cells(line_dots, cell_groups, row_assignments, row_centers, metrics)
        cells = [cell for cell in cells if len(cell.dots) >= self.min_dots_per_cell]
        cells = [cell for cell in cells if self._is_valid_cell(cell, metrics)]
        if not cells and len(line_dots) >= 2:
            cells = self._fallback_single_cell(line_dots, row_centers, metrics)
        cells.sort(key=lambda cell: cell.x)

        for index, cell in enumerate(cells, start=1):
            cell.cell_id = index

        confidence = self._segmentation_confidence(cells, line_dots)
        stability = self._stability_score(cells, metrics, row_centers)

        return SegmentationResult(
            cells=cells,
            confidence=confidence,
            stability_score=stability,
            row_centers=row_centers,
            col_spacing=metrics.col_spacing,
            row_spacing=metrics.row_spacing,
            cell_spacing=metrics.cell_spacing,
        )

    def _empty_result(self) -> SegmentationResult:
        return SegmentationResult(
            cells=[],
            confidence=0.0,
            stability_score=0.0,
            row_centers=(0.0, 0.0, 0.0),
            col_spacing=0.0,
            row_spacing=0.0,
            cell_spacing=0.0,
        )

    def _fit_row_model(
        self, dots: Sequence[DetectedDot]
    ) -> Tuple[Tuple[float, float, float], float, List[int]]:
        """Find the best-fitting 3-row Braille band (fast k-means + small refinement)."""
        if len(dots) <= 2:
            ys = [dot.y for dot in dots]
            center = float(np.mean(ys)) if ys else 0.0
            return (center, center, center), 20.0, list(range(len(dots)))

        ys = np.array([dot.y for dot in dots], dtype=np.float32).reshape(-1, 1)
        kmeans_centers = self._kmeans_1d(ys, k=self.row_count)
        base_spacing = (
            float(np.mean(np.diff(kmeans_centers)))
            if kmeans_centers.size > 1
            else 20.0
        )
        base_spacing = float(np.clip(base_spacing, 14.0, 36.0))
        base_y0 = float(kmeans_centers[0])

        best_centers = tuple(float(c) for c in kmeans_centers[: self.row_count])
        best_spacing = base_spacing
        best_indices = list(range(len(dots)))
        best_score = -1.0

        spacing_offsets = (-4.0, -2.0, 0.0, 2.0, 4.0)
        y0_offsets = (-6.0, -3.0, 0.0, 3.0, 6.0)

        for spacing_delta in spacing_offsets:
            spacing = base_spacing + spacing_delta
            if spacing < 12.0:
                continue
            tolerance = spacing * 0.42
            centers_arr = np.array(
                [base_y0, base_y0 + spacing, base_y0 + 2.0 * spacing],
                dtype=np.float32,
            )
            for y0_delta in y0_offsets:
                trial_centers = centers_arr + float(y0_delta)
                distances = np.abs(ys - trial_centers.reshape(1, -1))
                min_dist = distances.min(axis=1)
                labels = distances.argmin(axis=1)
                mask = min_dist <= tolerance
                if not np.any(mask):
                    continue

                indices = np.where(mask)[0].tolist()
                row_hits = np.bincount(labels[mask], minlength=3)
                score = float(len(indices)) + float(np.sum(row_hits > 0)) * 0.5

                if score > best_score:
                    best_score = score
                    best_centers = tuple(float(c) for c in trial_centers)
                    best_spacing = spacing
                    best_indices = indices

        expand_tolerance = best_spacing * 0.62
        centers_np = np.array(best_centers, dtype=np.float32)
        distances = np.abs(ys.reshape(-1, 1) - centers_np.reshape(1, -1))
        expanded = np.where(distances.min(axis=1) <= expand_tolerance)[0].tolist()
        if len(expanded) >= self.min_dots_per_cell:
            best_indices = expanded

        y_band_min = min(best_centers) - best_spacing * 0.55
        y_band_max = max(best_centers) + best_spacing * 0.55
        band_indices = [
            index
            for index, dot in enumerate(dots)
            if y_band_min <= dot.y <= y_band_max
        ]
        if len(band_indices) > len(best_indices):
            best_indices = band_indices

        if len(best_indices) < self.min_dots_per_cell:
            return (
                tuple(float(c) for c in kmeans_centers[: self.row_count]),
                base_spacing,
                list(range(len(dots))),
            )

        return best_centers, best_spacing, best_indices

    def _assign_rows(
        self,
        dots: Sequence[DetectedDot],
        row_centers: Tuple[float, float, float],
        row_spacing: float,
    ) -> List[int]:
        tolerance = max(row_spacing * self.row_merge_ratio, 5.0)
        assignments: List[int] = []
        for dot in dots:
            distances = [abs(dot.y - center) for center in row_centers]
            row = int(np.argmin(distances))
            if distances[row] > tolerance * 2.0:
                row = int(np.argmin(distances))
            assignments.append(row)
        return assignments

    def _estimate_spacing(
        self,
        dots: Sequence[DetectedDot],
        row_centers: Tuple[float, float, float],
        row_spacing: float,
    ) -> _SpacingMetrics:
        xs = np.array([dot.x for dot in dots], dtype=np.float32)
        radii = np.array([dot.radius for dot in dots], dtype=np.float32)
        avg_radius = float(np.mean(radii))
        row_assignments = self._assign_rows(dots, row_centers, row_spacing)
        row_tolerance = max(row_spacing * 0.45, avg_radius)

        col_candidates: List[float] = []

        for i in range(len(dots)):
            for j in range(i + 1, len(dots)):
                if row_assignments[i] != row_assignments[j]:
                    continue
                dx = abs(float(xs[i] - xs[j]))
                if dx <= avg_radius * 1.2:
                    continue
                if dx <= row_spacing * 1.5:
                    col_candidates.append(dx)

        col_spacing = self._robust_peak(
            [value for value in col_candidates if 8.0 <= value <= 40.0],
            default=max(avg_radius * 2.0, 12.0),
        )

        xs_sorted = np.sort(xs)
        x_gaps = np.diff(xs_sorted) if xs_sorted.size > 1 else np.array([], dtype=np.float32)
        if x_gaps.size:
            small_gaps = x_gaps[x_gaps <= avg_radius * 2.5]
            col_spacing = float(np.percentile(small_gaps, 50)) if small_gaps.size else col_spacing
            col_spacing = max(col_spacing, avg_radius * 1.5)

            inter_cell = x_gaps[x_gaps > col_spacing * 1.05]
            if inter_cell.size:
                cell_spacing = self._robust_peak(inter_cell.tolist(), default=col_spacing * 2.2)
            else:
                cell_spacing = float(np.percentile(x_gaps, 85))
            cell_spacing = max(cell_spacing, col_spacing * 1.8)
        else:
            cell_spacing = col_spacing * 1.55

        if cell_spacing <= col_spacing * 1.15:
            cell_spacing = col_spacing * 1.8
        if cell_spacing > col_spacing * 3.5:
            cell_spacing = col_spacing * 2.2

        return _SpacingMetrics(
            row_spacing=row_spacing,
            col_spacing=col_spacing,
            cell_spacing=cell_spacing,
            avg_radius=avg_radius,
        )

    def _robust_peak(self, values: Sequence[float], default: float) -> float:
        if not values:
            return default

        arr = np.array(values, dtype=np.float32)
        q25, q75 = np.percentile(arr, [25, 75])
        iqr = max(float(q75 - q25), 1.0)
        filtered = arr[(arr >= q25 - 1.5 * iqr) & (arr <= q75 + 1.5 * iqr)]
        if filtered.size == 0:
            filtered = arr

        bins = min(20, max(5, filtered.size // 2))
        hist, edges = np.histogram(filtered, bins=bins)
        peak_index = int(np.argmax(hist))
        center = float((edges[peak_index] + edges[peak_index + 1]) / 2.0)
        return center if center > 0 else default

    def _kmeans_1d(self, values: np.ndarray, k: int, max_iterations: int = 40) -> np.ndarray:
        values = values.reshape(-1, 1).astype(np.float32)
        k = max(1, min(k, values.shape[0]))
        if k == 1:
            return np.array([float(np.mean(values))], dtype=np.float32)

        sorted_values = values[:, 0]
        indices = np.linspace(0, sorted_values.size - 1, k, dtype=int)
        centers = sorted_values[indices].astype(np.float32).copy()

        for _ in range(max_iterations):
            distances = np.abs(values - centers.reshape(1, -1))
            labels = np.argmin(distances, axis=1)
            new_centers = np.zeros(k, dtype=np.float32)
            for cluster in range(k):
                members = values[labels == cluster, 0]
                new_centers[cluster] = float(np.mean(members)) if members.size else centers[cluster]
            if np.allclose(new_centers, centers, atol=0.5):
                break
            centers = new_centers

        return np.sort(centers)

    def _group_rows(
        self,
        dots: Sequence[DetectedDot],
        row_spacing: float,
    ) -> Tuple[List[int], Tuple[float, float, float]]:
        ys = np.array([[dot.y] for dot in dots], dtype=np.float32)
        centers = self._kmeans_1d(ys, k=self.row_count)
        row_centers = tuple(float(value) for value in centers[: self.row_count])

        while len(row_centers) < self.row_count:
            row_centers = row_centers + (row_centers[-1] + row_spacing,)

        assignments: List[int] = []
        for dot in dots:
            distances = [abs(dot.y - center) for center in row_centers]
            assignments.append(int(np.argmin(distances)))

        return assignments, row_centers  # type: ignore[return-value]

    def _group_cells(
        self,
        dots: Sequence[DetectedDot],
        row_assignments: Sequence[int],
        metrics: _SpacingMetrics,
    ) -> List[List[int]]:
        xs = np.array([dot.x for dot in dots], dtype=np.float32)
        if xs.size == 0:
            return []

        split_threshold = max(metrics.cell_spacing * 0.55, metrics.col_spacing * 1.15)
        sorted_indices = np.argsort(xs)
        gap_groups: List[List[int]] = []
        current = [int(sorted_indices[0])]

        for index in sorted_indices[1:]:
            index = int(index)
            previous = current[-1]
            if float(xs[index] - xs[previous]) > split_threshold:
                gap_groups.append(current)
                current = [index]
            else:
                current.append(index)
        gap_groups.append(current)

        return gap_groups

    def _merge_singleton_groups(
        self,
        dots: Sequence[DetectedDot],
        groups: List[List[int]],
        metrics: _SpacingMetrics,
    ) -> List[List[int]]:
        """Merge single-dot cells into the nearest neighbor within one cell width."""
        if len(groups) <= 1:
            return groups

        merge_distance = max(metrics.cell_spacing * 0.65, metrics.col_spacing * 1.8)
        merged = [list(group) for group in groups]
        changed = True

        while changed:
            changed = False
            singles = [index for index, group in enumerate(merged) if len(group) == 1]
            if not singles:
                break

            for single_index in singles:
                if single_index >= len(merged) or len(merged[single_index]) != 1:
                    continue

                dot = dots[merged[single_index][0]]
                best_target = -1
                best_distance = merge_distance

                for target_index, target_group in enumerate(merged):
                    if target_index == single_index or not target_group:
                        continue
                    target_x = float(np.mean([dots[i].x for i in target_group]))
                    distance = abs(dot.x - target_x)
                    if distance < best_distance:
                        best_distance = distance
                        best_target = target_index

                if best_target >= 0:
                    merged[best_target].extend(merged[single_index])
                    merged[single_index] = []
                    changed = True

            merged = [group for group in merged if group]

        return merged

    def _merge_close_values(self, values: Sequence[float], threshold: float) -> List[float]:
        if not values:
            return []

        sorted_values = sorted(values)
        merged = [sorted_values[0]]
        for value in sorted_values[1:]:
            if value - merged[-1] <= threshold:
                merged[-1] = (merged[-1] + value) / 2.0
            else:
                merged.append(value)
        return merged

    def _assign_grid_positions(
        self,
        cell_dots: Sequence[DetectedDot],
        row_assignments: Sequence[int],
        dot_indices: Sequence[int],
        row_centers: Tuple[float, float, float],
        metrics: _SpacingMetrics,
        col_split: float,
    ) -> Tuple[Tuple[bool, bool, bool, bool, bool, bool], Dict[int, DetectedDot]]:
        pattern = [False] * 6
        grid_map: Dict[int, DetectedDot] = {}

        xs = np.array([dot.x for dot in cell_dots], dtype=np.float32)
        col_tolerance = max(metrics.col_spacing * self.col_merge_ratio, metrics.avg_radius)
        x_span = float(xs.max() - xs.min()) if xs.size else 0.0
        single_column = x_span < max(metrics.col_spacing * 0.85, metrics.avg_radius * 2.5)

        if single_column and len(cell_dots) >= 2:
            ordered = sorted(zip(cell_dots, dot_indices), key=lambda item: item[0].y)
            for row_index, (dot, dot_index) in enumerate(ordered):
                row = min(row_index, self.row_count - 1)
                col = 0 if dot.x <= col_split else 1
                grid_index = row + col * 3
                if 0 <= grid_index <= 5:
                    grid_map[grid_index] = dot
                    pattern[grid_index] = True
            return tuple(pattern), grid_map

        for dot, dot_index in zip(cell_dots, dot_indices):
            row = row_assignments[dot_index]
            col = 0 if dot.x <= col_split else 1
            if abs(dot.x - col_split) <= col_tolerance * 0.2:
                left_count = int(np.sum(xs <= col_split))
                col = 0 if left_count >= len(xs) / 2 else 1

            grid_index = row + col * 3
            if grid_index < 0 or grid_index > 5:
                continue

            if grid_index in grid_map:
                existing = grid_map[grid_index]
                existing_dist = abs(existing.y - row_centers[row])
                new_dist = abs(dot.y - row_centers[row])
                if new_dist >= existing_dist:
                    continue

            grid_map[grid_index] = dot
            pattern[grid_index] = True

        return tuple(pattern), grid_map

    def _build_cells(
        self,
        dots: Sequence[DetectedDot],
        cell_groups: Sequence[Sequence[int]],
        row_assignments: Sequence[int],
        row_centers: Tuple[float, float, float],
        metrics: _SpacingMetrics,
    ) -> List[BrailleCell]:
        cells: List[BrailleCell] = []
        line_xs = np.array([dot.x for dot in dots], dtype=np.float32)
        left_anchor = float(np.percentile(line_xs, 8)) if line_xs.size else 0.0
        global_col_split = left_anchor + metrics.col_spacing * 0.55

        for group_index, dot_indices in enumerate(cell_groups, start=1):
            cell_dots = [dots[i] for i in dot_indices]
            pattern, grid_map = self._assign_grid_positions(
                cell_dots,
                row_assignments,
                dot_indices,
                row_centers,
                metrics,
                col_split=global_col_split,
            )

            xs = [dot.x for dot in cell_dots]
            ys = [dot.y for dot in cell_dots]
            padding = metrics.avg_radius * self.bbox_padding

            x_min = int(min(xs) - padding)
            y_min = int(min(ys) - padding)
            x_max = int(max(xs) + padding)
            y_max = int(max(ys) + padding)

            cells.append(
                BrailleCell(
                    cell_id=group_index,
                    dots=cell_dots,
                    pattern=pattern,
                    confidence=self._cell_confidence(cell_dots, pattern),
                    bounding_box=(x_min, y_min, x_max - x_min, y_max - y_min),
                    grid_map=grid_map,
                )
            )

        return cells

    def _fallback_single_cell(
        self,
        dots: Sequence[DetectedDot],
        row_centers: Tuple[float, float, float],
        metrics: _SpacingMetrics,
    ) -> List[BrailleCell]:
        """One Braille letter filling the frame (e.g. tight camera crop)."""
        dot_indices = list(range(len(dots)))
        row_assignments = self._assign_rows(dots, row_centers, metrics.row_spacing)
        line_xs = np.array([dot.x for dot in dots], dtype=np.float32)
        left_anchor = float(np.percentile(line_xs, 8)) if line_xs.size else 0.0
        col_split = left_anchor + metrics.col_spacing * 0.55
        pattern, grid_map = self._assign_grid_positions(
            dots,
            row_assignments,
            dot_indices,
            row_centers,
            metrics,
            col_split=col_split,
        )
        xs = [dot.x for dot in dots]
        ys = [dot.y for dot in dots]
        padding = metrics.avg_radius * self.bbox_padding
        cell = BrailleCell(
            cell_id=1,
            dots=list(dots),
            pattern=pattern,
            confidence=self._cell_confidence(dots, pattern),
            bounding_box=(
                int(min(xs) - padding),
                int(min(ys) - padding),
                int(max(xs) + padding) - int(min(xs) - padding),
                int(max(ys) + padding) - int(min(ys) - padding),
            ),
            grid_map=grid_map,
        )
        if sum(pattern) >= 2:
            return [cell]
        return []

    def _is_valid_cell(self, cell: BrailleCell, metrics: _SpacingMetrics) -> bool:
        filled = sum(cell.pattern)
        if filled == 0 or filled > self.max_dots_per_cell:
            return False
        if len(cell.dots) > self.max_dots_per_cell:
            return False

        _, certainty, fuzzy = match_pattern(cell.pattern)
        if filled == 1 and fuzzy and certainty < 0.4:
            return False
        if filled >= 2:
            return True

        if cell.width > metrics.cell_spacing * 2.2:
            return False
        if cell.height > metrics.row_spacing * 2.4:
            return False

        if filled >= 3:
            char, certainty, _ = match_pattern(cell.pattern)
            if char == "?" and certainty < 0.25:
                best_distance = min(
                    hamming_distance(cell.pattern, candidate) for candidate in BRAILLE_MAP
                )
                if best_distance > 2:
                    return False
        return True

    def _cell_confidence(
        self,
        cell_dots: Sequence[DetectedDot],
        pattern: Tuple[bool, bool, bool, bool, bool, bool],
    ) -> float:
        if not cell_dots:
            return 0.0

        dot_confidence = float(np.mean([dot.confidence for dot in cell_dots]))
        filled = sum(pattern)
        fill_ratio = filled / 6.0
        structure_bonus = 10.0 if filled >= 2 else 0.0
        sparse_penalty = 12.0 if len(cell_dots) < self.prefer_min_dots else 0.0
        left_col = sum(pattern[0:3])
        right_col = sum(pattern[3:6])
        grid_bonus = 8.0 if left_col > 0 and right_col > 0 else 0.0
        return round(
            min(
                dot_confidence + structure_bonus + grid_bonus + fill_ratio * 5.0 - sparse_penalty,
                100.0,
            ),
            1,
        )

    def _segmentation_confidence(
        self,
        cells: Sequence[BrailleCell],
        dots: Sequence[DetectedDot],
    ) -> float:
        if not cells:
            return 0.0

        cell_conf = float(np.mean([cell.confidence for cell in cells]))
        assigned = sum(len(cell.dots) for cell in cells)
        coverage = assigned / max(len(dots), 1)
        spacing_penalty = 0.0
        if len(cells) >= 2:
            left_edges = sorted(cell.x for cell in cells)
            gaps = np.diff(left_edges)
            if gaps.size:
                spacing_penalty = float(np.std(gaps) / max(np.mean(gaps), 1.0)) * 12.0
        score = cell_conf * 0.6 + coverage * 100.0 * 0.25 + (100.0 - spacing_penalty) * 0.15
        return round(min(max(score, 0.0), 100.0), 1)

    def _stability_score(
        self,
        cells: Sequence[BrailleCell],
        metrics: _SpacingMetrics,
        row_centers: Tuple[float, float, float],
    ) -> float:
        if len(cells) == 1 and sum(cells[0].pattern) >= 2:
            return 55.0
        if len(cells) < 2:
            return 50.0 if cells else 0.0

        cell_widths = [cell.width for cell in cells]
        cell_heights = [cell.height for cell in cells]
        left_edges = sorted(cell.x for cell in cells)
        spacings = np.diff(left_edges)

        width_cv = float(np.std(cell_widths) / max(np.mean(cell_widths), 1.0))
        height_cv = float(np.std(cell_heights) / max(np.mean(cell_heights), 1.0))
        spacing_error = (
            float(np.mean(np.abs(spacings - metrics.cell_spacing) / max(metrics.cell_spacing, 1.0)))
            if spacings.size
            else 0.5
        )
        row_gaps = np.diff(row_centers)
        row_error = (
            float(np.mean(np.abs(row_gaps - metrics.row_spacing) / max(metrics.row_spacing, 1.0)))
            if row_gaps.size
            else 0.5
        )

        penalty = width_cv * 25.0 + height_cv * 25.0 + spacing_error * 25.0 + row_error * 25.0
        return round(max(0.0, min(100.0, 100.0 - penalty)), 1)

    def format_pattern(self, pattern: Tuple[bool, bool, bool, bool, bool, bool]) -> str:
        return "[" + ", ".join("1" if value else "0" for value in pattern) + "]"
