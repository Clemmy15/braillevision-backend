"""Perspective correction and row alignment for Braille images."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detection import DetectedDot


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def auto_perspective_correct(image: np.ndarray) -> np.ndarray:
    """Flatten skewed page region when a quadrilateral boundary is found."""
    if len(image.shape) == 2:
        work = image.copy()
    else:
        work = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = work.shape[:2]
    blurred = cv2.GaussianBlur(work, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    contour = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(contour) / float(height * width)
    if area_ratio < 0.15:
        return image

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4:
        return image

    points = approx.reshape(4, 2).astype(np.float32)
    ordered = _order_points(points)
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    if len(image.shape) == 2:
        return cv2.warpPerspective(image, matrix, (width, height))
    return cv2.warpPerspective(image, matrix, (width, height))


def estimate_skew_angle(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=50, minLineLength=40, maxLineGap=15)
    if lines is None:
        return 0.0

    angles: List[float] = []
    for segment in lines[:60]:
        x1, y1, x2, y2 = segment[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180
        if abs(angle) < 25:
            angles.append(angle)

    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew_image(image: np.ndarray, max_angle: float = 12.0) -> np.ndarray:
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew_angle(gray)
    angle = float(np.clip(angle, -max_angle, max_angle))
    if abs(angle) < 0.4:
        return image

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def deskew_dots(dots: List[DetectedDot], angle_deg: float) -> List[DetectedDot]:
    if abs(angle_deg) < 0.3 or not dots:
        return dots

    xs = np.array([dot.x for dot in dots], dtype=np.float32)
    ys = np.array([dot.y for dot in dots], dtype=np.float32)
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    angle = np.radians(-angle_deg)
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))

    aligned: List[DetectedDot] = []
    for dot in dots:
        dx, dy = dot.x - cx, dot.y - cy
        rx = int(cx + dx * cos_a - dy * sin_a)
        ry = int(cy + dx * sin_a + dy * cos_a)
        aligned.append(
            DetectedDot(
                x=rx,
                y=ry,
                radius=dot.radius,
                confidence=dot.confidence,
                area=dot.area,
            )
        )
    return aligned
