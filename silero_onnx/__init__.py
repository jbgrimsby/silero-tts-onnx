"""silero_onnx: torch-free Silero TTS v5 RU (text -> ONNX -> WAV).

Public API is intentionally tiny; everything else is internal::

    from silero_onnx import SileroTTS
    tts = SileroTTS(model_dir="path/to/models")
    audio = tts.synthesize("Привет, мир!", speaker="xenia")  # float32 mono, 48kHz
"""
from .engine import SileroTTS, main

__all__ = ["SileroTTS", "main"]
