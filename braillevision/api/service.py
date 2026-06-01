"""API service layer — wraps core BrailleVision pipeline."""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from braillevision.core.debug_accuracy import annotate_confidence_panel, save_accuracy_debug
from braillevision.core.decoder import BrailleDecoder
from braillevision.core.detection import DotDetector
from braillevision.core.preprocessing import Preprocessor
from braillevision.core.quality import QualityAnalyzer, QualityStatus
from braillevision.core.segmentation import CellSegmenter
from braillevision.core.realtime import RealtimeAssistant
from braillevision.core.validation import compute_realistic_confidence, stabilize_text
from braillevision.core.visualization import Visualizer


@dataclass
class _Pipeline:
    preprocessor: Preprocessor
    detector: DotDetector
    segmenter: CellSegmenter
    decoder: BrailleDecoder
    quality_analyzer: QualityAnalyzer
    visualizer: Visualizer


_PIPELINE: Optional[_Pipeline] = None
_GUIDANCE = RealtimeAssistant()
_LAST_STABLE: Dict[str, Any] = {"signature": "", "text": "", "confidence": 0.0}


def _get_pipeline() -> _Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = _Pipeline(
            preprocessor=Preprocessor(),
            detector=DotDetector(),
            segmenter=CellSegmenter(prefer_min_dots=2),
            decoder=BrailleDecoder(),
            quality_analyzer=QualityAnalyzer(),
            visualizer=Visualizer(),
        )
    return _PIPELINE


def _quality_label(status: QualityStatus) -> str:
    labels = {
        QualityStatus.GOOD: "GOOD",
        QualityStatus.TOO_BLURRY: "BLURRY",
        QualityStatus.TOO_DARK: "TOO DARK",
        QualityStatus.TOO_BRIGHT: "TOO BRIGHT",
        QualityStatus.MOVE_CLOSER: "MOVE CLOSER",
    }
    return labels.get(status, "LOW")


def _image_to_base64_jpeg(image: np.ndarray, quality: int = 82, max_width: int = 960) -> str:
    output = image
    height, width = output.shape[:2]
    if width > max_width:
        scale = max_width / width
        output = cv2.resize(
            output,
            (max_width, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    success, buffer = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Failed to encode image.")
    return base64.b64encode(buffer).decode("ascii")


def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image file. Upload a valid JPEG or PNG.")
    return image


def _cells_signature(cells) -> str:
    parts = []
    for cell in sorted(cells, key=lambda item: item.x):
        parts.append("".join("1" if value else "0" for value in cell.pattern))
    return "|".join(parts)


def _stabilize_decode(text: str, confidence: float, cells) -> tuple[str, float]:
    """Reduce flicker; only cache high-quality reads."""
    global _LAST_STABLE
    signature = _cells_signature(cells)
    unknown_ratio = text.count("?") / max(len(text), 1)

    if confidence >= 62.0 and unknown_ratio < 0.15 and text.strip():
        _LAST_STABLE = {"signature": signature, "text": text, "confidence": confidence}
        return text, confidence

    if signature and signature == _LAST_STABLE.get("signature") and _LAST_STABLE.get("text"):
        prev_conf = float(_LAST_STABLE.get("confidence", 0.0))
        if confidence < prev_conf + 6.0:
            return str(_LAST_STABLE["text"]), prev_conf

    if confidence < 35.0 and _LAST_STABLE.get("text"):
        return str(_LAST_STABLE["text"]), float(_LAST_STABLE["confidence"])

    return text, confidence


def _build_speech_guidance(
    gray,
    binary,
    dot_count: int,
    decoded_text: str,
    confidence: float,
) -> str:
    """Natural-language prompts for text-to-speech accessibility."""
    guidance = _GUIDANCE.analyze(gray, binary, dot_count)
    text = decoded_text.strip()

    if not guidance.is_good:
        hint = guidance.message.rstrip(".")
        if text and confidence >= 55.0:
            return f"{hint}. I read: {text}."
        if text:
            return (
                f"{hint}. I think it says {text}, "
                "but I am not confident. Please try again."
            )
        return f"{hint}. I could not read the Braille clearly. Please try again."

    if text and confidence >= 55.0:
        return f"Good capture. I read: {text}."
    if text:
        return (
            f"I read {text}, but the scan was uncertain. "
            "Hold steady, move a little closer, and try again."
        )
    return (
        "The image looks okay, but I could not detect readable Braille. "
        "Move closer, improve the lighting, and scan again."
    )


def _debug_enabled(debug: Optional[bool]) -> bool:
    if debug is not None:
        return debug
    return os.environ.get("BRAILLEVISION_DEBUG", "").lower() in ("1", "true", "yes")


def process_image_array(
    image: np.ndarray,
    debug_accuracy: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run the full BrailleVision pipeline on a BGR OpenCV image."""
    start = time.perf_counter()
    pipeline = _get_pipeline()
    debug = _debug_enabled(debug_accuracy)

    result = pipeline.preprocessor.process(image)
    dots = pipeline.detector.detect(
        result.binary,
        grayscale=result.grayscale,
        enhanced=result.enhanced,
    )
    quality = pipeline.quality_analyzer.analyze(
        result.grayscale, result.binary, dots_detected=len(dots)
    )
    quality_code = _quality_label(quality.status)

    detection_confidence = pipeline.detector.overall_confidence(dots)
    segmentation = pipeline.segmenter.segment(dots, skew_angle=result.skew_angle)

    decode_result = pipeline.decoder.decode_cells(
        segmentation.cells,
        cell_spacing=segmentation.cell_spacing,
        verbose=False,
    )

    confidence_breakdown = compute_realistic_confidence(
        dots=dots,
        cells=segmentation.cells,
        characters=decode_result.characters,
        segmentation=segmentation,
        decoded_text=decode_result.decoded_text,
    )
    confidence = confidence_breakdown.overall

    text, confidence = stabilize_text(
        decode_result.decoded_text,
        confidence,
        segmentation.cells,
        segmentation_stability=segmentation.stability_score,
    )
    if not text.strip() and decode_result.decoded_text.strip():
        text = decode_result.decoded_text.strip()

    text, confidence = _stabilize_decode(text, confidence, segmentation.cells)

    char_labels = [
        character.character if character.character != " " else "_"
        for character in decode_result.characters
    ]

    annotated = pipeline.visualizer.draw_segmentation(
        result.original,
        segmentation.cells,
        character_labels=char_labels,
        dots=dots,
    )
    annotated = pipeline.visualizer.draw_summary(
        annotated,
        dot_count=len(dots),
        overall_confidence=detection_confidence,
        quality_message=quality.message,
        is_good=quality.is_acceptable,
        cell_count=len(segmentation.cells),
        segmentation_confidence=segmentation.confidence,
        stability_score=segmentation.stability_score,
        decoded_text=text,
        decode_confidence=confidence,
    )
    annotated = annotate_confidence_panel(annotated, confidence_breakdown)

    if debug:
        save_accuracy_debug(
            original=result.original,
            binary=result.binary,
            accepted=dots,
            rejected=pipeline.detector.last_debug.rejected,
            cells=segmentation.cells,
            row_centers=segmentation.row_centers,
            confidence=confidence_breakdown,
            output_dir=Path("output/debug_accuracy"),
        )

    guidance_speech = _build_speech_guidance(
        result.grayscale,
        result.binary,
        len(dots),
        text,
        confidence,
    )

    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 1)
    fps_estimate = round(min(1000.0 / max(elapsed_ms, 1.0), 60.0), 1)

    response: Dict[str, Any] = {
        "text": text,
        "confidence": round(confidence, 1),
        "quality": quality_code,
        "quality_message": quality.message,
        "guidance_speech": guidance_speech,
        "annotated_image": _image_to_base64_jpeg(annotated),
        "original_image": _image_to_base64_jpeg(result.original),
        "processing_time_ms": elapsed_ms,
        "fps_estimate": fps_estimate,
        "dots_detected": len(dots),
        "cells_detected": len(segmentation.cells),
        "segmentation_confidence": round(segmentation.confidence, 1),
        "stability_score": round(segmentation.stability_score, 1),
        "detection_confidence": round(detection_confidence, 1),
        "confidence_breakdown": confidence_breakdown.to_dict(),
    }
    return response


def process_uploaded_image(
    image_bytes: bytes,
    debug_accuracy: Optional[bool] = None,
) -> Dict[str, Any]:
    image = _bytes_to_bgr(image_bytes)
    return process_image_array(image, debug_accuracy=debug_accuracy)


def reset_decode_cache() -> None:
    global _LAST_STABLE
    _LAST_STABLE = {"signature": "", "text": "", "confidence": 0.0}
