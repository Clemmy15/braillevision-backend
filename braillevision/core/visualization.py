"""Visual overlays for BrailleVision — presentation-ready style."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detection import DetectedDot
from .segmentation import BrailleCell, GRID_LABELS


class Visualizer:
    """Draw detection and segmentation results on images."""

    DOT_RING = (72, 220, 120)
    DOT_CORE = (40, 120, 255)
    CELL_FILL = (255, 170, 60)
    CELL_BORDER = (255, 210, 120)
    CHAR_COLOR = (255, 255, 255)
    LABEL_BG = (20, 24, 38)

    def draw_segmentation(
        self,
        image: np.ndarray,
        cells: Sequence[BrailleCell],
        show_grid_positions: bool = False,
        character_labels: Optional[List[str]] = None,
        dots: Optional[Sequence[DetectedDot]] = None,
    ) -> np.ndarray:
        output = image.copy()
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        overlay = output.copy()

        for index, cell in enumerate(cells):
            x, y, w, h = cell.bounding_box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), self.CELL_FILL, -1)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), self.CELL_BORDER, 2)

            char_label = ""
            if character_labels and index < len(character_labels):
                char_label = character_labels[index]
                display = char_label if char_label.strip() and char_label != "_" else "?"
                label_size = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
                lx = x + (w - label_size[0]) // 2
                ly = max(y - 10, 22)
                cv2.rectangle(
                    overlay,
                    (lx - 6, ly - label_size[1] - 6),
                    (lx + label_size[0] + 6, ly + 4),
                    self.LABEL_BG,
                    -1,
                )
                cv2.putText(
                    overlay,
                    display,
                    (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    self.CHAR_COLOR,
                    2,
                    cv2.LINE_AA,
                )

            conf_label = f"{cell.confidence:.0f}%"
            cv2.putText(
                overlay,
                conf_label,
                (x + 4, y + h - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, 0.35, output, 0.65, 0, output)

        draw_dots = list(dots) if dots else []
        if not draw_dots:
            for cell in cells:
                draw_dots.extend(cell.dots)

        for dot in draw_dots:
            center = (dot.x, dot.y)
            radius = max(int(dot.radius), 4)
            cv2.circle(output, center, radius + 2, self.DOT_RING, 2, cv2.LINE_AA)
            cv2.circle(output, center, radius, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(output, center, max(2, radius // 2), self.DOT_CORE, -1, cv2.LINE_AA)

        return output

    def draw_quality_banner(
        self,
        image: np.ndarray,
        message: str,
        is_good: bool,
    ) -> np.ndarray:
        output = image.copy()
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        color = (80, 220, 120) if is_good else (80, 100, 255)
        cv2.rectangle(output, (0, 0), (output.shape[1], 40), (18, 20, 28), -1)
        cv2.putText(
            output,
            message[:70],
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        return output

    def draw_summary(
        self,
        image: np.ndarray,
        dot_count: int,
        overall_confidence: float,
        quality_message: str,
        is_good: bool,
        cell_count: int = 0,
        segmentation_confidence: float = 0.0,
        stability_score: float = 0.0,
        decoded_text: str = "",
        decode_confidence: float = 0.0,
    ) -> np.ndarray:
        output = self.draw_quality_banner(image, quality_message, is_good)
        summary = (
            f"Dots {dot_count} | Cells {cell_count} | "
            f"Det {overall_confidence:.0f}% | Seg {segmentation_confidence:.0f}% | "
            f"Decode {decode_confidence:.0f}%"
        )
        cv2.putText(
            output,
            summary,
            (10, output.shape[0] - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        if decoded_text:
            text_line = f'"{decoded_text[:52]}{"..." if len(decoded_text) > 52 else ""}"'
            cv2.putText(
                output,
                text_line,
                (10, output.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (160, 255, 180),
                1,
                cv2.LINE_AA,
            )
        return output
