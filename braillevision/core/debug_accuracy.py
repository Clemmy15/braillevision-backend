"""Optional debug artifacts for accuracy tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .detection import DetectedDot
from .segmentation import BrailleCell
from .validation import ConfidenceBreakdown

DEFAULT_DEBUG_DIR = Path("output/debug_accuracy")


def save_accuracy_debug(
    original: np.ndarray,
    binary: np.ndarray,
    accepted: Sequence[DetectedDot],
    rejected: Sequence[DetectedDot],
    cells: Sequence[BrailleCell],
    row_centers: Sequence[float],
    confidence: ConfidenceBreakdown,
    output_dir: Optional[Path] = None,
) -> Path:
    folder = output_dir or DEFAULT_DEBUG_DIR
    folder.mkdir(parents=True, exist_ok=True)

    overlay = original.copy()
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    for center_y in row_centers:
        y = int(center_y)
        cv2.line(overlay, (0, y), (overlay.shape[1], y), (255, 200, 0), 1)

    for dot in rejected:
        cv2.circle(overlay, (dot.x, dot.y), int(dot.radius), (0, 0, 255), 1)

    for dot in accepted:
        cv2.circle(overlay, (dot.x, dot.y), int(dot.radius), (0, 255, 0), 2)

    for cell in cells:
        x, y, w, h = cell.bounding_box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 1)

    cv2.imwrite(str(folder / "01_binary.jpg"), binary)
    cv2.imwrite(str(folder / "02_dots_overlay.jpg"), overlay)

    breakdown_path = folder / "confidence_breakdown.txt"
    lines = [f"{key}: {value}" for key, value in confidence.to_dict().items()]
    breakdown_path.write_text("\n".join(lines), encoding="utf-8")

    return folder


def annotate_confidence_panel(
    image: np.ndarray,
    breakdown: ConfidenceBreakdown,
) -> np.ndarray:
    panel = image.copy()
    lines = [
        f"Overall: {breakdown.overall:.0f}%",
        f"Dots: {breakdown.dot_quality:.0f}%",
        f"Seg fit: {breakdown.segmentation_fit:.0f}%",
        f"Stability: {breakdown.segmentation_stability:.0f}%",
        f"Decode: {breakdown.decode_certainty:.0f}%",
        f"Geometry: {breakdown.geometric_consistency:.0f}%",
        f"Text: {breakdown.text_validity:.0f}%",
        f"Penalties: -{breakdown.penalties:.0f}",
    ]
    y = 24
    for line in lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 40), 1)
        y += 20
    return panel
