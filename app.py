"""FastAPI backend for BrailleVision."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from braillevision.api.service import (
    process_uploaded_image,
    reset_decode_cache,
)

SAMPLE_PATH = PROJECT_ROOT / "samples" / "braille_test.jpg"

app = FastAPI(
    title="BrailleVision API",
    description="REST API for physical Braille detection and decoding.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str = "ok"


class ProcessImageResponse(BaseModel):
    text: str = ""
    confidence: float = Field(0.0, ge=0.0, le=100.0)
    quality: str = "LOW"
    quality_message: str = ""
    annotated_image: str = ""
    original_image: str = ""
    processing_time_ms: float = 0.0
    fps_estimate: float = 0.0
    dots_detected: int = 0
    cells_detected: int = 0
    segmentation_confidence: float = 0.0
    stability_score: float = 0.0
    detection_confidence: float = 0.0


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/demo", response_model=ProcessImageResponse)
def demo_scan() -> ProcessImageResponse:
    """Run pipeline on bundled sample — ideal for hackathon demos."""
    if not SAMPLE_PATH.exists():
        raise HTTPException(status_code=404, detail="Demo sample image not found.")
    reset_decode_cache()
    result = process_uploaded_image(SAMPLE_PATH.read_bytes())
    return ProcessImageResponse(**result)


@app.post("/process-image", response_model=ProcessImageResponse)
async def process_image(file: UploadFile = File(...)) -> ProcessImageResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.).")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        result = process_uploaded_image(image_bytes)
        return ProcessImageResponse(**result)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Processing failed: {error}") from error
