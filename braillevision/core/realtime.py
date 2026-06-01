"""Real-time webcam BrailleVision pipeline with accessibility assistant."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .decoder import BrailleDecoder, DecodeResult
from .validation import compute_realistic_confidence, stabilize_text
from .detection import DetectedDot, DotDetector
from .preprocessing import Preprocessor, PreprocessResult
from .quality import QualityAnalyzer, QualityStatus
from .segmentation import BrailleCell, CellSegmenter, SegmentationResult
from .speech import SpeechEngine


class ScanMode(str, Enum):
    IDLE = "Idle"
    SCANNING = "Scanning"
    GOOD_CAPTURE = "Good Capture"
    LOW_QUALITY = "Low Quality"


@dataclass
class AssistantGuidance:
    """Live accessibility guidance for the user."""

    message: str
    quality_label: str
    blur_score: float
    brightness: float
    dot_density: float
    skew_degrees: float
    is_good: bool
    suggest_straighten: bool = False


@dataclass
class FrameResult:
    """Output of processing a single frame."""

    preprocess: PreprocessResult
    dots: List[DetectedDot]
    segmentation: SegmentationResult
    decode: DecodeResult
    guidance: AssistantGuidance
    scan_mode: ScanMode
    detection_confidence: float
    frame_index: int = 0
    was_processed: bool = True


@dataclass
class StabilityState:
    """Smoothed decode output to prevent flicker."""

    stable_text: str = ""
    stable_confidence: float = 0.0
    candidate_text: str = ""
    candidate_confidence: float = 0.0
    candidate_streak: int = 0
    last_cells_signature: str = ""
    last_speech_time: float = 0.0
    last_spoken_text: str = ""

    def reset(self) -> None:
        self.stable_text = ""
        self.stable_confidence = 0.0
        self.candidate_text = ""
        self.candidate_confidence = 0.0
        self.candidate_streak = 0
        self.last_cells_signature = ""
        self.last_spoken_text = ""


class RealtimeAssistant:
    """Camera guidance: blur, brightness, dot density, skew."""

    def __init__(
        self,
        blur_threshold: float = 80.0,
        dark_threshold: float = 60.0,
        bright_threshold: float = 240.0,
        min_dot_density: float = 0.00004,
        far_dot_density: float = 0.000012,
        skew_threshold_deg: float = 6.0,
    ) -> None:
        self.quality = QualityAnalyzer(
            blur_threshold=blur_threshold,
            dark_threshold=dark_threshold,
            bright_threshold=bright_threshold,
        )
        self.min_dot_density = min_dot_density
        self.far_dot_density = far_dot_density
        self.skew_threshold_deg = skew_threshold_deg

    def measure_dot_density(self, binary: np.ndarray, dot_count: int) -> float:
        area = float(binary.shape[0] * binary.shape[1])
        return dot_count / area if area > 0 else 0.0

    def estimate_skew_degrees(self, binary: np.ndarray) -> float:
        edges = cv2.Canny(binary, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180.0, threshold=40, minLineLength=30, maxLineGap=12
        )
        if lines is None or len(lines) == 0:
            return 0.0

        angles: List[float] = []
        for segment in lines[:40]:
            x1, y1, x2, y2 = segment[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            while angle > 90:
                angle -= 180
            while angle < -90:
                angle += 180
            if abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return 0.0
        return float(np.median(angles))

    def analyze(
        self,
        gray: np.ndarray,
        binary: np.ndarray,
        dot_count: int,
    ) -> AssistantGuidance:
        report = self.quality.analyze(gray, binary)
        dot_density = self.measure_dot_density(binary, dot_count)
        skew = self.estimate_skew_degrees(binary)
        suggest_straighten = abs(skew) > self.skew_threshold_deg

        message = report.message
        quality_label = "GOOD"
        is_good = report.is_acceptable

        if report.status == QualityStatus.TOO_DARK:
            message = "Increase lighting"
            quality_label = "LOW"
            is_good = False
        elif report.status == QualityStatus.TOO_BRIGHT:
            message = "Reduce glare or lighting"
            quality_label = "LOW"
            is_good = False
        elif report.status == QualityStatus.TOO_BLURRY:
            message = "Hold camera steady — image blurry"
            quality_label = "BLURRY"
            is_good = False
        elif dot_density < self.far_dot_density:
            message = "Too far from page — move camera closer"
            quality_label = "LOW"
            is_good = False
        elif dot_density < self.min_dot_density or report.status == QualityStatus.MOVE_CLOSER:
            message = "Move camera closer"
            quality_label = "LOW"
            is_good = False
        elif suggest_straighten:
            message = "Tilt camera to straighten page"
            quality_label = "LOW"
            is_good = False
        elif is_good:
            message = "Good capture quality"
            quality_label = "GOOD"

        return AssistantGuidance(
            message=message,
            quality_label=quality_label,
            blur_score=report.blur_score,
            brightness=report.brightness,
            dot_density=dot_density,
            skew_degrees=skew,
            is_good=is_good,
            suggest_straighten=suggest_straighten,
        )


class StabilityTracker:
    """Frame smoothing for decoded text and speech gating."""

    def __init__(
        self,
        confidence_threshold: float = 60.0,
        streak_required: int = 3,
        speech_cooldown_sec: float = 5.0,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.streak_required = streak_required
        self.speech_cooldown_sec = speech_cooldown_sec
        self.state = StabilityState()

    def cells_signature(self, cells: List[BrailleCell]) -> str:
        parts = []
        for cell in sorted(cells, key=lambda item: item.x):
            parts.append("".join("1" if value else "0" for value in cell.pattern))
        return "|".join(parts)

    def update(self, decode: DecodeResult, cells: List[BrailleCell]) -> Tuple[str, float, bool]:
        """
        Update stable text. Returns (stable_text, confidence, speech_triggered).
        speech_triggered is True when a new stable sentence should be spoken.
        """
        signature = self.cells_signature(cells)
        text = decode.decoded_text.strip()
        confidence = decode.overall_confidence

        if signature == self.state.last_cells_signature and text == self.state.candidate_text:
            if confidence >= self.confidence_threshold:
                self.state.candidate_streak += 1
            else:
                self.state.candidate_streak = max(0, self.state.candidate_streak - 1)
        else:
            self.state.candidate_text = text
            self.state.candidate_confidence = confidence
            self.state.candidate_streak = 1 if confidence >= self.confidence_threshold else 0
            self.state.last_cells_signature = signature

        speech_triggered = False
        if (
            self.state.candidate_streak >= self.streak_required
            and confidence >= self.confidence_threshold
            and text
        ):
            if text != self.state.stable_text:
                self.state.stable_text = text
                self.state.stable_confidence = confidence
                now = time.time()
                if (
                    text != self.state.last_spoken_text
                    and now - self.state.last_speech_time >= self.speech_cooldown_sec
                ):
                    speech_triggered = True
                    self.state.last_speech_time = now
                    self.state.last_spoken_text = text

        return self.state.stable_text, self.state.stable_confidence, speech_triggered

    def reset(self) -> None:
        self.state.reset()


class RealtimePipeline:
    """Runs BrailleVision on webcam frames with optional frame skipping."""

    def __init__(
        self,
        frame_skip: int = 2,
        stability_threshold: float = 60.0,
        speech_cooldown: float = 5.0,
        speech_rate: int = 160,
    ) -> None:
        self.frame_skip = max(1, frame_skip)
        self.preprocessor = Preprocessor()
        self.detector = DotDetector()
        self.segmenter = CellSegmenter()
        self.decoder = BrailleDecoder()
        self.assistant = RealtimeAssistant()
        self.stability = StabilityTracker(
            confidence_threshold=stability_threshold,
            speech_cooldown_sec=speech_cooldown,
        )
        self.speech = SpeechEngine(rate=speech_rate, enabled=True)

        self._frame_count = 0
        self._last_result: Optional[FrameResult] = None
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    def toggle_pause(self) -> bool:
        self._paused = not self._paused
        return self._paused

    def reset(self) -> None:
        self.stability.reset()
        self._last_result = None
        self._frame_count = 0

    def should_process_frame(self) -> bool:
        if self._paused:
            return False
        return self._frame_count % self.frame_skip == 0

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        self._frame_count += 1

        if not self.should_process_frame() and self._last_result is not None:
            cached = self._last_result
            return FrameResult(
                preprocess=cached.preprocess,
                dots=cached.dots,
                segmentation=cached.segmentation,
                decode=cached.decode,
                guidance=cached.guidance,
                scan_mode=cached.scan_mode,
                detection_confidence=cached.detection_confidence,
                frame_index=self._frame_count,
                was_processed=False,
            )

        preprocess = self.preprocessor.process(frame)
        dots = self.detector.detect(
            preprocess.binary,
            grayscale=preprocess.grayscale,
            enhanced=preprocess.enhanced,
        )
        detection_confidence = self.detector.overall_confidence(dots)
        guidance = self.assistant.analyze(
            preprocess.grayscale, preprocess.binary, len(dots)
        )

        if not dots:
            empty_decode = DecodeResult(
                raw_cells=[],
                decoded_text="",
                character_confidences=[],
                overall_confidence=0.0,
            )
            segmentation = SegmentationResult(
                cells=[],
                confidence=0.0,
                stability_score=0.0,
                row_centers=(0.0, 0.0, 0.0),
                col_spacing=0.0,
                row_spacing=0.0,
                cell_spacing=0.0,
            )
            scan_mode = ScanMode.IDLE if guidance.is_good else ScanMode.LOW_QUALITY
        else:
            segmentation = self.segmenter.segment(dots, skew_angle=preprocess.skew_angle)
            empty_decode = None
            if segmentation.cells:
                decode = self.decoder.decode_cells(
                    segmentation.cells,
                    cell_spacing=segmentation.cell_spacing,
                    verbose=False,
                )
                breakdown = compute_realistic_confidence(
                    dots=dots,
                    cells=segmentation.cells,
                    characters=decode.characters,
                    segmentation=segmentation,
                    decoded_text=decode.decoded_text,
                )
                stable_text, stable_conf = stabilize_text(
                    decode.decoded_text,
                    breakdown.overall,
                    segmentation.cells,
                    segmentation_stability=segmentation.stability_score,
                )
                decode.decoded_text = stable_text
                decode.overall_confidence = stable_conf
                scan_mode = (
                    ScanMode.GOOD_CAPTURE
                    if guidance.is_good and stable_conf >= 58
                    else ScanMode.SCANNING
                )
                if not guidance.is_good:
                    scan_mode = ScanMode.LOW_QUALITY
            else:
                decode = DecodeResult(
                    raw_cells=[],
                    decoded_text="",
                    character_confidences=[],
                    overall_confidence=0.0,
                )
                scan_mode = ScanMode.SCANNING if guidance.is_good else ScanMode.LOW_QUALITY

        if empty_decode is not None:
            decode = empty_decode

        result = FrameResult(
            preprocess=preprocess,
            dots=dots,
            segmentation=segmentation,
            decode=decode,
            guidance=guidance,
            scan_mode=scan_mode,
            detection_confidence=detection_confidence,
            frame_index=self._frame_count,
            was_processed=True,
        )
        self._last_result = result
        return result

    def speak_async(self, text: str) -> bool:
        if not text.strip():
            return False

        def _run() -> None:
            self.speech.speak(text)

        threading.Thread(target=_run, daemon=True).start()
        return True

    def force_speak(self) -> bool:
        text = self.stability.state.stable_text or self.stability.state.candidate_text
        return self.speak_async(text)


class RealtimeRenderer:
    """Draw live overlay on webcam frames."""

    BANNER_HEIGHT = 72
    FOOTER_HEIGHT = 56

    def render(
        self,
        frame: np.ndarray,
        result: FrameResult,
        stable_text: str,
        stable_confidence: float,
        fps: float,
        paused: bool,
    ) -> np.ndarray:
        output = frame.copy()
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        self._draw_detections(output, result)
        self._draw_banner(output, result, fps, paused)
        self._draw_footer(output, stable_text, stable_confidence, result)
        self._draw_confidence_meter(output, stable_confidence)
        return output

    def _draw_detections(self, output: np.ndarray, result: FrameResult) -> None:
        for dot in result.dots:
            cv2.circle(output, (dot.x, dot.y), max(int(dot.radius), 3), (0, 255, 0), 2)
            cv2.circle(output, (dot.x, dot.y), 2, (0, 0, 255), -1)

        char_labels = [
            character.character if character.character != " " else "_"
            for character in result.decode.characters
        ]

        for index, cell in enumerate(result.segmentation.cells):
            x, y, w, h = cell.bounding_box
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 180, 0), 2)
            if index < len(char_labels):
                label = char_labels[index]
                cv2.putText(
                    output,
                    label,
                    (x + w // 2 - 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

    def _draw_banner(
        self,
        output: np.ndarray,
        result: FrameResult,
        fps: float,
        paused: bool,
    ) -> None:
        h, w = output.shape[:2]
        cv2.rectangle(output, (0, 0), (w, self.BANNER_HEIGHT), (25, 25, 25), -1)

        mode = result.scan_mode.value
        if paused:
            mode = "PAUSED"

        mode_color = {
            ScanMode.GOOD_CAPTURE.value: (80, 220, 80),
            ScanMode.SCANNING.value: (0, 200, 255),
            ScanMode.LOW_QUALITY.value: (60, 60, 255),
            ScanMode.IDLE.value: (180, 180, 180),
            "PAUSED": (200, 200, 100),
        }.get(mode, (200, 200, 200))

        cv2.putText(
            output,
            f"BrailleVision Assistant | Mode: {mode}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            mode_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"Quality: {result.guidance.quality_label} | {result.guidance.message}",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (w - 90, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def _draw_footer(
        self,
        output: np.ndarray,
        stable_text: str,
        stable_confidence: float,
        result: FrameResult,
    ) -> None:
        h, w = output.shape[:2]
        y0 = h - self.FOOTER_HEIGHT
        cv2.rectangle(output, (0, y0), (w, h), (25, 25, 25), -1)

        display_text = stable_text or result.decode.decoded_text or "(scanning...)"
        cv2.putText(
            output,
            f'Text: "{display_text[:60]}"',
            (10, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"Live: {result.decode.decoded_text[:40] or '-'} | "
            f"Conf: {stable_confidence:.0f}% (stable) / {result.decode.overall_confidence:.0f}% (frame)",
            (10, y0 + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def _draw_confidence_meter(self, output: np.ndarray, confidence: float) -> None:
        h, w = output.shape[:2]
        bar_x = w - 30
        bar_y = self.BANNER_HEIGHT + 10
        bar_h = h - self.BANNER_HEIGHT - self.FOOTER_HEIGHT - 20
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + 16, bar_y + bar_h), (60, 60, 60), -1)
        fill_h = int(bar_h * (confidence / 100.0))
        color = (80, 220, 80) if confidence >= 60 else (0, 180, 255) if confidence >= 35 else (60, 60, 255)
        cv2.rectangle(
            output,
            (bar_x, bar_y + bar_h - fill_h),
            (bar_x + 16, bar_y + bar_h),
            color,
            -1,
        )
        cv2.putText(
            output,
            f"{confidence:.0f}",
            (bar_x - 4, bar_y + bar_h + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )


def run_webcam_pipeline(
    camera_index: int = 0,
    frame_skip: int = 2,
    width: int = 640,
    height: int = 480,
    stability_threshold: float = 60.0,
    speech_cooldown: float = 5.0,
    speech_rate: int = 160,
) -> int:
    """Main real-time loop — OpenCV window with accessibility assistant."""
    print("=" * 50)
    print("BrailleVision - Real-Time Assistant Mode")
    print("=" * 50)
    print("Controls: [S] Speak  [P] Pause  [R] Reset  [Q] Quit")
    print("Camera started")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    pipeline = RealtimePipeline(
        frame_skip=frame_skip,
        stability_threshold=stability_threshold,
        speech_cooldown=speech_cooldown,
        speech_rate=speech_rate,
    )
    renderer = RealtimeRenderer()

    window_name = "BrailleVision - Live Assistant"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    fps = 0.0
    frames_logged = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame read failed.")
                break

            result = pipeline.process_frame(frame)
            stable_text = pipeline.stability.state.stable_text
            stable_conf = pipeline.stability.state.stable_confidence

            if result.was_processed:
                frames_logged += 1
                if frames_logged % 10 == 0:
                    print(
                        f"Frame processed | Quality: {result.guidance.quality_label} | "
                        f"Dots: {len(result.dots)} | Cells: {len(result.segmentation.cells)}"
                    )

                stable_text, stable_conf, speech_now = pipeline.stability.update(
                    result.decode, result.segmentation.cells
                )

                if (
                    speech_now
                    or (
                        stable_text
                        and pipeline.stability.state.candidate_streak
                        >= pipeline.stability.streak_required
                        and frames_logged % 20 == 0
                    )
                ) and stable_text:
                    print(f"Stable text detected: {stable_text}")

                if speech_now and stable_text:
                    print("Speech triggered")
                    print(f"Quality: {result.guidance.quality_label}")
                    pipeline.speak_async(stable_text)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - prev_time, 0.001)
            prev_time = now

            display = renderer.render(
                frame,
                result,
                stable_text=stable_text,
                stable_confidence=stable_conf,
                fps=fps,
                paused=pipeline.paused,
            )

            cv2.putText(
                display,
                "[S]peak [P]ause [R]eset [Q]uit",
                (10, display.shape[0] - renderer.FOOTER_HEIGHT - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (160, 160, 160),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("p"), ord("P")):
                paused = pipeline.toggle_pause()
                print("Paused" if paused else "Resumed")
            if key in (ord("r"), ord("R")):
                pipeline.reset()
                print("Reset — stability and text cleared")
            if key in (ord("s"), ord("S")):
                if pipeline.force_speak():
                    print("Manual speech triggered")
                else:
                    print("Nothing to speak yet")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nCamera stopped.")
        print("Real-time session ended.")

    return 0
