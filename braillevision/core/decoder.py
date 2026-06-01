"""Map Braille dot patterns to English text (Grade 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .segmentation import BrailleCell

from .braille_map import BRAILLE_MAP, CHAR_TO_BRAILLE, Pattern, hamming_distance, match_pattern

PatternInput = Union[Pattern, Sequence[int], Sequence[bool]]


@dataclass
class DecodedCharacter:
    """A single decoded Braille character."""

    character: str
    confidence: float
    cell_id: int
    pattern: Pattern
    is_unknown: bool = False
    is_fuzzy: bool = False
    match_certainty: float = 1.0

    def to_dict(self) -> dict:
        return {
            "character": self.character,
            "confidence": round(self.confidence, 1),
            "cell_id": self.cell_id,
            "dots": [int(value) for value in self.pattern],
        }


@dataclass
class DecodeResult:
    """Full decoding output for a scan."""

    raw_cells: List[dict]
    decoded_text: str
    character_confidences: List[float]
    characters: List[DecodedCharacter] = field(default_factory=list)
    overall_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "raw_cells": self.raw_cells,
            "decoded_text": self.decoded_text,
            "character_confidences": [round(value, 1) for value in self.character_confidences],
            "overall_confidence": round(self.overall_confidence, 1),
            "characters": [character.to_dict() for character in self.characters],
        }


class BrailleDecoder:
    """Convert binary Braille patterns into English text."""

    def __init__(
        self,
        segmentation_weight: float = 0.6,
        detection_weight: float = 0.4,
        space_gap_ratio: float = 1.35,
        unknown_penalty: float = 25.0,
    ) -> None:
        self.segmentation_weight = segmentation_weight
        self.detection_weight = detection_weight
        self.space_gap_ratio = space_gap_ratio
        self.unknown_penalty = unknown_penalty

    @staticmethod
    def normalize_pattern(pattern: PatternInput) -> Pattern:
        values = tuple(bool(int(value)) for value in pattern)
        if len(values) != 6:
            raise ValueError(f"Braille pattern must have 6 elements, got {len(values)}")
        return values  # type: ignore[return-value]

    def decode_pattern(self, pattern: PatternInput) -> str:
        normalized = self.normalize_pattern(pattern)
        character, _, _ = match_pattern(normalized)
        return character

    def _dot_detection_confidence(self, cell: BrailleCell) -> float:
        if not cell.dots:
            return 0.0
        return float(sum(dot.confidence for dot in cell.dots) / len(cell.dots))

    def _character_confidence(
        self,
        cell: BrailleCell,
        character: str,
        match_certainty: float,
        is_fuzzy: bool,
    ) -> float:
        segmentation_conf = cell.confidence
        detection_conf = self._dot_detection_confidence(cell)
        combined = (
            self.segmentation_weight * segmentation_conf
            + self.detection_weight * detection_conf
        )
        combined *= match_certainty
        if is_fuzzy:
            combined *= 0.82
        if character == "?":
            combined = max(0.0, combined - self.unknown_penalty)
        return round(min(combined, 100.0), 1)

    def _gap_is_word_space(
        self,
        previous: BrailleCell,
        current: BrailleCell,
        cell_spacing: Optional[float],
    ) -> bool:
        if cell_spacing is None or cell_spacing <= 0:
            cell_spacing = max(previous.width, current.width, 20)

        prev_right = previous.x + previous.width
        gap = current.x - prev_right
        return gap > cell_spacing * self.space_gap_ratio

    def decode_cells(
        self,
        cells: Sequence[BrailleCell],
        cell_spacing: Optional[float] = None,
        verbose: bool = True,
    ) -> DecodeResult:
        """Decode segmented cells into text with spacing and confidence."""
        if verbose:
            print("\nDecoding Braille cells...")

        raw_cells = [cell.to_dict() for cell in cells]
        characters: List[DecodedCharacter] = []
        text_parts: List[str] = []

        sorted_cells = sorted(cells, key=lambda cell: cell.x)

        for index, cell in enumerate(sorted_cells):
            if index > 0 and self._gap_is_word_space(sorted_cells[index - 1], cell, cell_spacing):
                if text_parts and text_parts[-1] != " ":
                    text_parts.append(" ")
                    if verbose:
                        print("  Word space detected (large cell gap)")

            pattern = self.normalize_pattern(cell.pattern)
            character, match_certainty, is_fuzzy = match_pattern(pattern)
            confidence = self._character_confidence(
                cell, character, match_certainty, is_fuzzy
            )
            is_unknown = character == "?" or match_certainty < 0.35

            decoded = DecodedCharacter(
                character=character,
                confidence=confidence,
                cell_id=cell.cell_id,
                pattern=pattern,
                is_unknown=is_unknown,
                is_fuzzy=is_fuzzy,
                match_certainty=match_certainty,
            )
            characters.append(decoded)
            text_parts.append(character)

            if verbose:
                display = "space" if character == " " else character
                print(f"Character mapped: {display}  (cell {cell.cell_id}, {confidence:.0f}%)")

        decoded_text = "".join(text_parts).strip()
        decoded_text = " ".join(decoded_text.split())

        character_confidences = [character.confidence for character in characters]
        overall_confidence = self._overall_confidence(characters)

        if verbose:
            print(f'Final output: "{decoded_text}"')
            print(f"Confidence: {overall_confidence:.0f}%")

        return DecodeResult(
            raw_cells=raw_cells,
            decoded_text=decoded_text,
            character_confidences=character_confidences,
            characters=characters,
            overall_confidence=overall_confidence,
        )

    def _overall_confidence(self, characters: Sequence[DecodedCharacter]) -> float:
        if not characters:
            return 0.0

        weights = []
        scores = []
        for character in characters:
            weight = 0.35 if character.is_unknown else 0.7 if character.is_fuzzy else 1.0
            weight *= character.match_certainty
            weights.append(weight)
            scores.append(character.confidence * weight)

        total_weight = sum(weights) or 1.0
        return round(sum(scores) / total_weight, 1)

    def reconstruct_text(
        self,
        characters: Sequence[DecodedCharacter],
        cell_spacing: Optional[float] = None,
        cells: Optional[Sequence[BrailleCell]] = None,
    ) -> str:
        """Merge decoded characters into a sentence (spacing from gaps or blank cells)."""
        if not characters:
            return ""

        if cells is not None:
            return self.decode_cells(cells, cell_spacing=cell_spacing, verbose=False).decoded_text

        return "".join(character.character for character in characters).strip()
