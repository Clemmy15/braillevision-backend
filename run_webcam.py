"""Launch BrailleVision real-time assistant mode."""

from braillevision.core.realtime import run_webcam_pipeline

if __name__ == "__main__":
    raise SystemExit(run_webcam_pipeline())
