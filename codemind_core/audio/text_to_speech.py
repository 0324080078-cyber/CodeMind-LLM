"""Coqui TTS — local, no cloud."""

import os
import time
from typing import Optional


class CoquiTTS:
    OUTPUT_DIR = "./outputs/audio"

    def __init__(self, model_name="tts_models/en/ljspeech/tacotron2-DDC"):
        self.model_name = model_name
        self.tts = None
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self._load()

    def _load(self):
        try:
            from TTS.api import TTS
            self.tts = TTS(model_name=self.model_name, progress_bar=False, gpu=False)
            print("Coqui TTS loaded")
        except ImportError:
            print("Install: pip install TTS")
        except Exception as e:
            print(f"TTS failed: {e}")

    def speak(self, text, output_path=None):
        if self.tts is None:
            return ""
        if not output_path:
            output_path = os.path.join(self.OUTPUT_DIR, f"tts_{int(time.time())}.wav")
        self.tts.tts_to_file(text=text, file_path=output_path)
        return output_path
