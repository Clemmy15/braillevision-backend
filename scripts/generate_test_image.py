"""Generate synthetic embossed Braille test images for pipeline development."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from braillevision.core.decoder import BRAILLE_MAP


def pattern_for_char(char: str) -> tuple[bool, bool, bool, bool, bool, bool] | None:
    target = char.lower()
    for pattern, mapped in BRAILLE_MAP.items():
        if mapped == target:
            return pattern
    return None


def draw_embossed_dot(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    light_direction: tuple[float, float] = (-0.6, -0.8),
) -> None:
    x, y = center
    height, width = image.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    mask = distance <= radius

    dx = (xx - x) / max(radius, 1)
    dy = (yy - y) / max(radius, 1)
    lx, ly = light_direction
    shading = 0.55 + 0.35 * (dx * lx + dy * ly)
    shading = np.clip(shading, 0.0, 1.0)

    bump = np.zeros_like(image, dtype=np.float32)
    inner = distance <= radius * 0.85
    bump[inner] = 18.0 * (1.0 - distance[inner] / max(radius, 1))
    bump[mask] += 8.0

    shadow = np.zeros_like(image, dtype=np.float32)
    shadow[distance <= radius * 1.2] = -12.0 * (1.0 - distance[distance <= radius * 1.2] / (radius * 1.2))

    base = image.astype(np.float32)
    base[mask] = np.clip(base[mask] + bump[mask] * shading[mask] + shadow[mask], 0, 255)
    image[:] = base.astype(np.uint8)


def render_word(
    text: str,
    dot_radius: int = 11,
    col_spacing: int = 22,
    row_spacing: int = 24,
    cell_spacing: int = 34,
    margin: int = 60,
    noise_level: float = 4.0,
) -> np.ndarray:
    patterns = []
    for char in text:
        pattern = pattern_for_char(char)
        if pattern is None:
            continue
        patterns.append(pattern)

    if not patterns:
        raise ValueError("No valid Braille characters in text.")

    height = margin * 2 + row_spacing * 2 + dot_radius * 2
    width = margin * 2 + len(patterns) * cell_spacing + dot_radius * 2
    image = np.full((height, width, 3), 210, dtype=np.uint8)

    for cell_index, pattern in enumerate(patterns):
        cell_x = margin + cell_index * cell_spacing
        cell_y = margin

        positions = [
            (cell_x, cell_y),
            (cell_x, cell_y + row_spacing),
            (cell_x, cell_y + row_spacing * 2),
            (cell_x + col_spacing, cell_y),
            (cell_x + col_spacing, cell_y + row_spacing),
            (cell_x + col_spacing, cell_y + row_spacing * 2),
        ]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for active, (x, y) in zip(pattern, positions):
            if active:
                draw_embossed_dot(gray, (x, y), dot_radius)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if noise_level > 0:
        noise = np.random.normal(0, noise_level, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Braille test images.")
    parser.add_argument("--text", default="hello", help="Word to render in Grade 1 Braille.")
    parser.add_argument(
        "--output",
        default="samples/braille_test.jpg",
        help="Output image path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = render_word(args.text)
    cv2.imwrite(str(output_path), image)
    print(f"Saved synthetic Braille image to {output_path}")


if __name__ == "__main__":
    main()
