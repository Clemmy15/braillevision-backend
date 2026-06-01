"""Grade 1 English Braille patterns and fuzzy matching."""

from __future__ import annotations

from typing import Dict, Tuple

Pattern = Tuple[bool, bool, bool, bool, bool, bool]

BRAILLE_MAP: Dict[Pattern, str] = {
    (True, False, False, False, False, False): "a",
    (True, True, False, False, False, False): "b",
    (True, False, False, True, False, False): "c",
    (True, False, False, True, True, False): "d",
    (True, False, False, False, True, False): "e",
    (True, True, False, True, False, False): "f",
    (True, True, False, True, True, False): "g",
    (True, True, False, False, True, False): "h",
    (False, True, False, True, False, False): "i",
    (False, True, False, True, True, False): "j",
    (True, False, True, False, False, False): "k",
    (True, True, True, False, False, False): "l",
    (True, False, True, True, False, False): "m",
    (True, False, True, True, True, False): "n",
    (True, False, True, False, True, False): "o",
    (True, True, True, True, False, False): "p",
    (True, True, True, True, True, False): "q",
    (True, True, True, False, True, False): "r",
    (False, True, True, True, False, False): "s",
    (False, True, True, True, True, False): "t",
    (True, False, True, False, False, True): "u",
    (True, True, True, False, False, True): "v",
    (False, True, False, True, True, True): "w",
    (True, False, True, True, False, True): "x",
    (True, False, True, True, True, True): "y",
    (True, False, True, False, True, True): "z",
    (False, False, False, False, False, False): " ",
    (False, True, False, False, True, True): ".",
    (False, False, True, False, False, True): ",",
    (False, False, True, True, False, False): ";",
    (False, False, True, True, True, False): ":",
    (False, False, True, False, True, False): "?",
    (True, False, False, False, True, True): "!",
    (False, True, False, True, False, True): "'",
    (True, False, False, False, False, True): "-",
    (False, False, True, True, False, True): "/",
    (True, True, False, False, False, True): '"',
}

CHAR_TO_BRAILLE: Dict[str, Pattern] = {char: pattern for pattern, char in BRAILLE_MAP.items()}


def hamming_distance(a: Pattern, b: Pattern) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def match_pattern(pattern: Pattern) -> Tuple[str, float, bool]:
    if pattern in BRAILLE_MAP:
        return BRAILLE_MAP[pattern], 1.0, False

    best_char = "?"
    best_distance = 7
    for candidate, char in BRAILLE_MAP.items():
        distance = hamming_distance(pattern, candidate)
        if distance < best_distance:
            best_distance = distance
            best_char = char

    if best_distance == 1:
        return best_char, 0.72, True
    if best_distance == 2:
        return best_char, 0.45, True
    return "?", 0.15, True
