"""HTTP API wrappers for BrailleVision."""

from .service import process_image_array, process_uploaded_image

__all__ = ["process_image_array", "process_uploaded_image"]
