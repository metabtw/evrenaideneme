"""Sessiz motor: TTS kapalıyken veya başsız testlerde kullanılır."""

from __future__ import annotations


class NullSpeechEngine:
    """Hiçbir şey yapmayan, her zaman başarılı görünen motor."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    @property
    def name(self) -> str:
        return "none"

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def stop(self) -> None:
        return None

    def is_busy(self) -> bool:
        return False

    def close(self) -> None:
        return None
