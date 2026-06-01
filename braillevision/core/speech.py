"""Text-to-speech output for recognized Braille text."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

DEFAULT_LAST_SCAN_PATH = Path("output/last_scan.txt")


class SpeechEngine:
    """Offline speech synthesis using pyttsx3 with replay support."""

    def __init__(
        self,
        rate: int = 160,
        volume: float = 1.0,
        enabled: bool = True,
        last_scan_path: Optional[Path] = None,
    ) -> None:
        self.rate = rate
        self.volume = volume
        self.enabled = enabled
        self.last_scan_path = last_scan_path or DEFAULT_LAST_SCAN_PATH
        self._engine = None
        self._last_text: str = ""
        self._load_persisted_scan()

    def _get_engine(self):
        if self._engine is None:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            self._engine.setProperty("volume", self.volume)
        return self._engine

    def set_rate(self, rate: int) -> None:
        """Adjust speech speed (words per minute, typical range 100-200)."""
        self.rate = max(80, min(300, rate))
        if self._engine is not None:
            self._engine.setProperty("rate", self.rate)

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, volume))
        if self._engine is not None:
            self._engine.setProperty("volume", self.volume)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    @property
    def last_text(self) -> str:
        return self._last_text

    def _load_persisted_scan(self) -> None:
        if self.last_scan_path.exists():
            self._last_text = self.last_scan_path.read_text(encoding="utf-8").strip()

    def _persist_last_scan(self, text: str) -> None:
        self.last_scan_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_scan_path.write_text(text, encoding="utf-8")

    def speak(self, text: str, store: bool = True) -> bool:
        """Speak text aloud. Returns True if speech was attempted."""
        cleaned = text.strip()
        if not cleaned:
            return False

        if store:
            self._last_text = cleaned
            self._persist_last_scan(cleaned)

        if not self.enabled:
            return False

        try:
            engine = self._get_engine()
            engine.setProperty("rate", self.rate)
            engine.say(cleaned)
            engine.runAndWait()
            return True
        except Exception as error:
            print(f"Speech unavailable: {error}")
            print("  Install pyttsx3: pip install pyttsx3")
            self.enabled = False
            return False

    def read_last_scan(self) -> bool:
        """Replay speech for the most recently decoded text."""
        if not self._last_text:
            print("No previous scan to read.")
            return False
        print(f'Reading last scan: "{self._last_text}"')
        return self.speak(self._last_text, store=False)

    def replay(self) -> bool:
        """Alias for read_last_scan (accessibility replay)."""
        return self.read_last_scan()
