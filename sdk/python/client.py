"""
CodeMind Python SDK
Official SDK for building on top of CodeMind AI Platform.
"""

import json
import os
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.parse


class CodeMindClient:
    """Python SDK for CodeMind AI Platform."""

    def __init__(self, api_key="", base_url="http://localhost:8000", timeout=120):
        self.api_key = api_key or os.environ.get("CODEMIND_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}", "User-Agent": "CodeMind-Python-SDK/2.0"}

    def _post(self, endpoint, data):
        req = urllib.request.Request(f"{self.base_url}{endpoint}", data=json.dumps(data).encode(), headers=self._headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def _get(self, endpoint):
        req = urllib.request.Request(f"{self.base_url}{endpoint}", headers=self._headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def chat(self, message, system=None, session_id=None, max_tokens=2048, temperature=0.7, history=None, tools=None):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": message})
        r = self._post("/v1/chat/completions", {"model": "codemind-v2", "messages": msgs, "max_tokens": max_tokens, "temperature": temperature, "session_id": session_id, "tools": tools})
        return r["choices"][0]["message"]["content"]

    def complete(self, prefix, suffix="", language="python", max_tokens=256, temperature=0.3):
        return self._post("/v1/ide/complete", {"prefix": prefix, "suffix": suffix, "language": language, "max_tokens": max_tokens, "temperature": temperature}).get("completion", "")

    def explain(self, code, language="python"):
        return self._post("/v1/ide/explain", {"code": code, "language": language}).get("explanation", "")

    def fix(self, code, error, language="python"):
        return self._post("/v1/ide/fix", {"code": code, "error": error, "language": language}).get("fixed_code", "")

    def document(self, code, language="python"):
        return self._post("/v1/ide/document", {"code": code, "language": language}).get("documented_code", "")

    def refactor(self, code, instruction, language="python"):
        return self._post("/v1/ide/refactor", {"code": code, "instruction": instruction, "language": language}).get("refactored_code", "")

    def test(self, code, framework="pytest", language="python"):
        return self._post("/v1/ide/test", {"code": code, "framework": framework, "language": language}).get("tests", "")

    def execute(self, code, language="python", stdin=None):
        return self._post("/v1/execute", {"code": code, "language": language, "stdin": stdin})

    def generate_image(self, prompt, negative_prompt="blurry, bad quality", size="512x512", steps=20, seed=None):
        r = self._post("/v1/images/generations", {"prompt": prompt, "negative_prompt": negative_prompt, "size": size, "steps": steps, "seed": seed, "n": 1})
        return r["data"][0].get("local_path", "")

    def transcribe(self, audio_path, language=None):
        return self._post("/v1/audio/transcriptions", {"audio_path": audio_path, "language": language}).get("text", "")

    def speak(self, text, output_path=None):
        return self._post("/v1/audio/speech", {"input": text, "output_path": output_path}).get("audio_path", "")

    def remember(self, content, session_id="global"):
        return self._post("/v1/memory/store", {"content": content, "session_id": session_id}).get("stored", False)

    def recall(self, query, session_id=None, top_k=5):
        return self._post("/v1/memory/recall", {"query": query, "session_id": session_id, "top_k": top_k}).get("memories", [])

    def search(self, query, max_results=5):
        return self._post("/v1/search", {"query": query, "max_results": max_results}).get("results", [])

    def health(self):
        return self._get("/health")

    def models(self):
        return self._get("/v1/models").get("data", [])
