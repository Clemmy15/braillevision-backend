"""BrailleVision demo entry point — full pipeline through speech."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from braillevision.core.debug_accuracy import annotate_confidence_panel, save_accuracy_debug
from braillevision.core.decoder import BrailleDecoder
from braillevision.core.detection import DotDetector
from braillevision.core.preprocessing import Preprocessor
from braillevision.core.quality import QualityAnalyzer
from braillevision.core.segmentation import CellSegmenter
from braillevision.core.validation import compute_realistic_confidence, stabilize_text
from braillevision.core.realtime import run_webcam_pipeline
from braillevision.core.speech import SpeechEngine
from braillevision.core.visualization import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BrailleVision - detect, segment, decode, and speak physical Braille."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="samples/braille_test.jpg",
        help="Path to input image (default: samples/braille_test.jpg).",
    )
    parser.add_argument(
        "--output",
        default="output/braillevision_result.jpg",
        help="Path to save annotated output image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display result in an OpenCV window.",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a synthetic Braille test image before processing.",
    )
    parser.add_argument(
        "--no-speak",
        action="store_true",
        help="Skip text-to-speech output.",
    )
    parser.add_argument(
        "--speech-rate",
        type=int,
        default=160,
        help="Speech speed in words per minute (default: 160).",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay speech from the last decoded scan (no image processing).",
    )
    parser.add_argument(
        "--save-json",
        default="output/decode_result.json",
        help="Path to save JSON decode result (default: output/decode_result.json).",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Start real-time Braille assistant mode using the webcam.",
    )
    parser.add_argument(
        "--debug-accuracy",
        action="store_true",
        help="Save accuracy debug artifacts to output/debug_accuracy/.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Process every Nth frame for performance (default: 2).",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=60.0,
        help="Minimum confidence %% to lock stable text (default: 60).",
    )
    return parser.parse_args()


def ensure_sample_image(path: Path) -> None:
    if path.exists():
        return

    from scripts.generate_test_image import render_word

    path.parent.mkdir(parents=True, exist_ok=True)
    image = render_word("hello")
    cv2.imwrite(str(path), image)
    print(f"Generated sample image: {path}")


def run_pipeline(
    image_path: Path,
    output_path: Path,
    show: bool,
    speak: bool,
    speech_rate: int,
    save_json_path: Path,
    debug_accuracy: bool = False,
) -> int:
    print("=" * 50)
    print("BrailleVision - Full Pipeline (Detect -> Decode -> Speak)")
    print("=" * 50)

    preprocessor = Preprocessor()
    detector = DotDetector()
    segmenter = CellSegmenter(prefer_min_dots=2)
    decoder = BrailleDecoder()
    quality_analyzer = QualityAnalyzer()
    visualizer = Visualizer()
    speech = SpeechEngine(rate=speech_rate, enabled=speak)

    print(f"\n[1/6] Loading image: {image_path}")
    image = preprocessor.load_image(str(image_path))

    print("[2/6] Preprocessing...")
    result = preprocessor.process(image)
    print("Preprocessing complete")
    print(f"  - Grayscale shape: {result.grayscale.shape}")

    print("\n[3/6] Analyzing capture quality...")
    quality = quality_analyzer.analyze(result.grayscale, result.binary)
    print(f"  - {quality}")

    print("\n[4/6] Detecting Braille dots...")
    dots = detector.detect(result.binary, grayscale=result.grayscale, enhanced=result.enhanced)
    detection_confidence = detector.overall_confidence(dots)
    print(f"Dots detected: {len(dots)}")
    print(f"Detection confidence: {detection_confidence:.1f}%")

    if not dots:
        print("\nNo dots detected.")
        return 1

    print("\n[5/6] Segmenting Braille cells...")
    segmentation = segmenter.segment(dots, skew_angle=result.skew_angle)
    print(f"Cells detected: {len(segmentation.cells)}")
    print(f"Segmentation confidence: {segmentation.confidence:.1f}%")
    print(f"Segmentation stability score: {segmentation.stability_score:.1f}%")

    for cell in segmentation.cells:
        print(f"Cell {cell.cell_id} dots: {segmenter.format_pattern(cell.pattern)}")

    if not segmentation.cells:
        print("\nNo valid Braille cells formed.")
        return 1

    print("\n[6/6] Decoding Braille to English...")
    decode_result = decoder.decode_cells(
        segmentation.cells,
        cell_spacing=segmentation.cell_spacing,
        verbose=True,
    )

    confidence_breakdown = compute_realistic_confidence(
        dots=dots,
        cells=segmentation.cells,
        characters=decode_result.characters,
        segmentation=segmentation,
        decoded_text=decode_result.decoded_text,
    )
    final_text, final_confidence = stabilize_text(
        decode_result.decoded_text,
        confidence_breakdown.overall,
        segmentation.cells,
        segmentation_stability=segmentation.stability_score,
    )
    print(f"\nRealistic confidence: {final_confidence:.1f}%")
    print(f"  Breakdown: {confidence_breakdown.to_dict()}")

    char_labels = [
        character.character if character.character != " " else "_"
        for character in decode_result.characters
    ]

    if speak:
        print("\nSpeaking decoded text...")
        speech.speak(final_text or decode_result.decoded_text)

    annotated = visualizer.draw_segmentation(
        result.original,
        segmentation.cells,
        character_labels=char_labels,
    )
    annotated = visualizer.draw_summary(
        annotated,
        dot_count=len(dots),
        overall_confidence=detection_confidence,
        quality_message=quality.message,
        is_good=quality.is_acceptable,
        cell_count=len(segmentation.cells),
        segmentation_confidence=segmentation.confidence,
        stability_score=segmentation.stability_score,
        decoded_text=final_text or decode_result.decoded_text,
        decode_confidence=final_confidence,
    )
    annotated = annotate_confidence_panel(annotated, confidence_breakdown)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    print(f"\nSaved annotated output: {output_path}")

    save_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = decode_result.to_dict()
    payload["realistic_confidence"] = final_confidence
    payload["confidence_breakdown"] = confidence_breakdown.to_dict()
    payload["decoded_text"] = final_text or decode_result.decoded_text
    with open(save_json_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)
    print(f"Saved decode JSON: {save_json_path}")

    debug_dir = output_path.parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "01_grayscale.jpg"), result.grayscale)
    cv2.imwrite(str(debug_dir / "02_enhanced.jpg"), result.enhanced)
    cv2.imwrite(str(debug_dir / "03_binary.jpg"), result.binary)
    cv2.imwrite(str(debug_dir / "06_final.jpg"), annotated)

    if debug_accuracy:
        save_accuracy_debug(
            original=result.original,
            binary=result.binary,
            accepted=dots,
            rejected=detector.last_debug.rejected,
            cells=segmentation.cells,
            row_centers=segmentation.row_centers,
            confidence=confidence_breakdown,
            output_dir=Path("output/debug_accuracy"),
        )
        print("Saved accuracy debug to output/debug_accuracy/")

    if show:
        cv2.imshow("BrailleVision - Result", annotated)
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    print(f'Text output: {final_text or decode_result.decoded_text}')
    print(f"Realistic confidence: {final_confidence:.1f}%")
    print("=" * 50)
    print("\nPipeline complete.")
    return 0


def main() -> int:
    args = parse_args()

    if args.webcam:
        return run_webcam_pipeline(
            camera_index=args.camera,
            frame_skip=args.frame_skip,
            stability_threshold=args.stability_threshold,
            speech_cooldown=5.0,
            speech_rate=args.speech_rate,
        )

    speech = SpeechEngine(rate=args.speech_rate, enabled=not args.no_speak)

    if args.replay:
        if speech.read_last_scan():
            return 0
        print("Run a scan first to populate the last decoded text.")
        return 1

    image_path = Path(args.image)
    if args.generate_sample or not image_path.exists():
        ensure_sample_image(image_path)

    if not image_path.exists():
        print(f"Error: image not found at {image_path}", file=sys.stderr)
        return 1

    return run_pipeline(
        image_path,
        Path(args.output),
        args.show,
        speak=not args.no_speak,
        speech_rate=args.speech_rate,
        save_json_path=Path(args.save_json),
        debug_accuracy=args.debug_accuracy,
    )


if __name__ == "__main__":
    raise SystemExit(main())
