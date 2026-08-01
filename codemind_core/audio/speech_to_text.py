"""Whisper STT — local, no cloud."""

import os
from typing import Optional


class WhisperSTT:
    def __init__(self, model_name="base"):
        self.model_name = model_name
        self.model = None
        self._load()

    def _load(self):
        try:
            import whisper
            self.model = whisper.load_model(self.model_name)
            print(f"Whisper {self.model_name} loaded")
        except ImportError:
            print("Install: pip install openai-whisper")
        except Exception as e:
            print(f"Whisper failed: {e}")

    def transcribe(self, audio_path, language=None):
        if self.model is None:
            return "Whisper not loaded."
        if not os.path.exists(audio_path):
            return f"File not found: {audio_path}"
        kwargs = {}
        if language:
            kwargs["language"] = language
        result = self.model.transcribe(audio_path, fp16=False, **kwargs)
        return result["text"].strip()
