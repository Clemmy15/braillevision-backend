"""Braille decode validation and realistic confidence scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .braille_map import Pattern, match_pattern
from .decoder import DecodedCharacter
from .detection import DetectedDot
from .segmentation import BrailleCell, SegmentationResult

COMMON_BIGRAMS = {
    "th", "he", "in", "er", "an", "re", "on", "at", "en", "es",
    "ha", "nd", "or", "nt", "ea", "ti", "to", "it", "st", "io",
}


@dataclass
class ConfidenceBreakdown:
    dot_quality: float = 0.0
    segmentation_stability: float = 0.0
    segmentation_fit: float = 0.0
    decode_certainty: float = 0.0
    geometric_consistency: float = 0.0
    text_validity: float = 0.0
    penalties: float = 0.0
    overall: float = 0.0

    def to_dict(self) -> dict:
        return {
            "dot_quality": round(self.dot_quality, 1),
            "segmentation_stability": round(self.segmentation_stability, 1),
            "segmentation_fit": round(self.segmentation_fit, 1),
            "decode_certainty": round(self.decode_certainty, 1),
            "geometric_consistency": round(self.geometric_consistency, 1),
            "text_validity": round(self.text_validity, 1),
            "penalties": round(self.penalties, 1),
            "overall": round(self.overall, 1),
        }


def _text_validity_score(text: str) -> float:
    if not text:
        return 0.0

    letters = re.sub(r"[^a-zA-Z]", "", text)
    if not letters:
        return 20.0 if text.strip() else 0.0

    unknown_ratio = text.count("?") / max(len(text), 1)
    letter_ratio = len(letters) / max(len(text.replace(" ", "")), 1)
    score = 100.0 * letter_ratio

    lower = text.lower()
    bigram_hits = sum(1 for index in range(len(lower) - 1) if lower[index : index + 2] in COMMON_BIGRAMS)
    bigram_ratio = bigram_hits / max(len(lower) - 1, 1)
    score = score * 0.7 + bigram_ratio * 100.0 * 0.3

    if re.search(r"[^a-zA-Z\s.,!?;:'\"-]", text):
        score *= 0.6
    score -= unknown_ratio * 80.0
    return max(0.0, min(100.0, score))


def _geometric_score(cells: Sequence[BrailleCell], segmentation: SegmentationResult) -> float:
    if not cells:
        return 0.0

    scores: List[float] = []
    for cell in cells:
        filled = sum(cell.pattern)
        if filled == 0:
            scores.append(10.0)
            continue
        left = sum(cell.pattern[0:3])
        right = sum(cell.pattern[3:6])
        balance = min(left, right) / max(left + right, 1)
        fill_score = min(filled / 4.0, 1.0) * 100.0
        scores.append(fill_score * 0.6 + balance * 100.0 * 0.4)

    base = float(np.mean(scores)) if scores else 0.0
    return base * 0.7 + segmentation.stability_score * 0.3


def compute_realistic_confidence(
    dots: Sequence[DetectedDot],
    cells: Sequence[BrailleCell],
    characters: Sequence[DecodedCharacter],
    segmentation: SegmentationResult,
    decoded_text: str,
) -> ConfidenceBreakdown:
    dot_quality = float(np.mean([dot.confidence for dot in dots])) if dots else 0.0

    incomplete_cells = sum(1 for cell in cells if sum(cell.pattern) < 2)
    incomplete_ratio = incomplete_cells / max(len(cells), 1)

    fuzzy_count = sum(1 for char in characters if getattr(char, "is_fuzzy", False))
    unknown_count = sum(1 for char in characters if char.is_unknown or char.character == "?")
    unknown_ratio = unknown_count / max(len(characters), 1)

    decode_certainty = 0.0
    if characters:
        certainties = [getattr(char, "match_certainty", 0.5) for char in characters]
        decode_certainty = float(np.mean(certainties)) * 100.0

    text_validity = _text_validity_score(decoded_text)
    geometric = _geometric_score(cells, segmentation)

    penalties = (
        unknown_ratio * 35.0
        + incomplete_ratio * 25.0
        + (fuzzy_count / max(len(characters), 1)) * 15.0
    )

    overall = (
        dot_quality * 0.18
        + segmentation.confidence * 0.14
        + segmentation.stability_score * 0.18
        + decode_certainty * 0.22
        + geometric * 0.14
        + text_validity * 0.14
        - penalties
    )

    if unknown_ratio > 0.35:
        overall = min(overall, 45.0)
    if decode_certainty < 50.0:
        overall = min(overall, 55.0)
    if not decoded_text.strip():
        overall = min(overall, 15.0)
    if segmentation.stability_score < 25.0:
        overall = min(overall, 48.0)
    if geometric < 35.0:
        overall = min(overall, 52.0)
    if len(cells) < 3 and decoded_text.strip():
        overall = min(overall, 40.0)

    overall = max(0.0, min(overall, 92.0))
    strong_signals = (
        unknown_ratio < 0.05
        and decode_certainty >= 85.0
        and segmentation.stability_score >= 55.0
        and geometric >= 50.0
        and text_validity >= 70.0
    )
    if not strong_signals:
        overall = min(overall, 78.0)

    return ConfidenceBreakdown(
        dot_quality=dot_quality,
        segmentation_stability=segmentation.stability_score,
        segmentation_fit=segmentation.confidence,
        decode_certainty=decode_certainty,
        geometric_consistency=geometric,
        text_validity=text_validity,
        penalties=penalties,
        overall=round(overall, 1),
    )


def stabilize_text(
    text: str,
    confidence: float,
    cells: Sequence[BrailleCell],
    min_confidence: float = 52.0,
    segmentation_stability: float = 100.0,
) -> Tuple[str, float]:
    """Prefer empty/short output over high-confidence garbage."""
    unknown_ratio = text.count("?") / max(len(text), 1)
    if unknown_ratio > 0.5:
        return "", max(0.0, confidence - 40.0)
    if confidence < min_confidence and unknown_ratio > 0.2:
        return "", confidence * 0.5
    if segmentation_stability < 20.0 and len(cells) < 5:
        confidence = min(confidence, 42.0)
    if text.strip() and unknown_ratio == 0.0 and len(cells) <= 2:
        return text, confidence
    return text, confidence
