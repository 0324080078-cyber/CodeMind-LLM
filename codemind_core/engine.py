"""
CodeMind Platform Engine
Zero external AI API dependencies.
Everything runs on YOUR hardware with YOUR models.
"""

import os
import torch
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("codemind.engine")


class CodeMindPlatform:
    VERSION = "2.0.0"

    def __init__(
        self,
        device="auto",
        enable_vision=True,
        enable_audio=True,
        enable_memory=True,
        enable_sandbox=True,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.llm = None
        self.tokenizer = None
        self.vision = None
        self.audio_stt = None
        self.audio_tts = None
        self.memory = None
        self.sandbox = None
        self.web_search = None
        self.router = None
        self.context_manager = None
        self.enable_vision = enable_vision
        self.enable_audio = enable_audio
        self.enable_memory = enable_memory
        self.enable_sandbox = enable_sandbox
        self._initialize()

    def _initialize(self):
        logger.info(f"CodeMind Platform v{self.VERSION} starting on {self.device}")
        self._load_llm()
        self._load_context_manager()
        self._load_router()
        if self.enable_vision:
            self._load_vision()
        if self.enable_audio:
            self._load_audio()
        if self.enable_memory:
            self._load_memory()
        if self.enable_sandbox:
            self._load_sandbox()
        self._load_web_search()
        logger.info("CodeMind Platform fully initialized")

    def _load_llm(self):
        try:
            from model import CodeMindLLM
            from tokenizer import CodeMindTokenizer
            checkpoint = "./checkpoints/checkpoint-best"
            tok_path = "./tokenizer/vocab/tokenizer.json"
            if os.path.exists(checkpoint):
                self.llm = CodeMindLLM.from_pretrained(checkpoint).to(self.device).eval()
                logger.info(f"LLM loaded from {checkpoint}")
            else:
                logger.warning(f"No checkpoint at {checkpoint}. Run: python codemind.py train")
            if os.path.exists(tok_path):
                self.tokenizer = CodeMindTokenizer(tok_path)
                logger.info(f"Tokenizer loaded. Vocab: {len(self.tokenizer)}")
        except Exception as e:
            logger.error(f"LLM load failed: {e}")

    def _load_context_manager(self):
        from codemind_core.context.manager import ContextManager
        self.context_manager = ContextManager(max_tokens=8192)

    def _load_router(self):
        from codemind_core.router.smart_router import SmartRouter
        self.router = SmartRouter(self)

    def _load_vision(self):
        try:
            from codemind_core.vision.generator import VisionEngine
            self.vision = VisionEngine(device=str(self.device))
        except Exception as e:
            logger.warning(f"Vision not available: {e}")

    def _load_audio(self):
        try:
            from codemind_core.audio.speech_to_text import WhisperSTT
            from codemind_core.audio.text_to_speech import CoquiTTS
            self.audio_stt = WhisperSTT()
            self.audio_tts = CoquiTTS()
        except Exception as e:
            logger.warning(f"Audio not available: {e}")

    def _load_memory(self):
        try:
            from codemind_core.memory.vector_memory import VectorMemory
            self.memory = VectorMemory(persist_path="./memory/chroma_db")
        except Exception as e:
            logger.warning(f"Memory not available: {e}")

    def _load_sandbox(self):
        from codemind_core.tools.sandbox import CodeSandbox
        self.sandbox = CodeSandbox()

    def _load_web_search(self):
        from codemind_core.tools.web_search import DuckDuckGoSearch
        self.web_search = DuckDuckGoSearch()

    def _generate(self, prompt, max_tokens=1024, temperature=0.7, top_p=0.95):
        if self.llm is None or self.tokenizer is None:
            return "Model not loaded. Run: python codemind.py train"
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.llm.generate(
                tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_ids = out[0][len(ids):].tolist()
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    async def chat(self, messages, session_id="default", max_tokens=2048, temperature=0.7, top_p=0.95, stream=False, tools=None):
        if self.llm is None:
            return self._error("Model not trained yet. Run: python codemind.py train")
        context = self.context_manager.get_context(session_id, messages)
        result = await self.router.route(context=context, tools=tools or [], max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        if self.memory and result.get("content"):
            self.memory.store(session_id=session_id, content=result["content"], metadata={"role": "assistant"})
        self.context_manager.update(session_id, result["content"])
        return {"id": result.get("id","cm"), "model": f"CodeMind-{self.VERSION}", "content": result["content"], "tokens_used": result.get("tokens_used",0), "tools_used": result.get("tools_used",[]), "finish_reason": "stop"}

    async def complete(self, prompt, max_tokens=1024, temperature=0.7, top_p=0.95, language=None):
        generated = self._generate(prompt, max_tokens, temperature, top_p)
        return {"prompt": prompt, "completion": generated, "language": language or "auto", "model": f"CodeMind-{self.VERSION}"}

    async def generate_image(self, prompt, negative_prompt="blurry, bad quality", width=512, height=512, steps=20, guidance_scale=7.5, seed=None):
        if self.vision is None:
            return self._error("Vision not available.")
        path = self.vision.generate(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, num_inference_steps=steps, guidance_scale=guidance_scale, seed=seed)
        return {"image_path": path, "prompt": prompt}

    async def transcribe_audio(self, audio_path, language=None):
        if self.audio_stt is None:
            return self._error("Audio STT not available.")
        text = self.audio_stt.transcribe(audio_path, language)
        return {"text": text, "audio_path": audio_path}

    async def speak(self, text, output_path=None):
        if self.audio_tts is None:
            return self._error("Audio TTS not available.")
        path = self.audio_tts.speak(text, output_path)
        return {"audio_path": path, "text": text}

    async def execute_code(self, code, language="python", stdin=None):
        if self.sandbox is None:
            return self._error("Sandbox not available.")
        return self.sandbox.execute(code=code, language=language, stdin=stdin)

    async def search_web(self, query, max_results=5):
        if self.web_search is None:
            return self._error("Web search not available.")
        results = self.web_search.search(query, max_results)
        return {"query": query, "results": results}

    async def remember(self, content, session_id="global"):
        if self.memory is None:
            return self._error("Memory not available.")
        self.memory.store(session_id=session_id, content=content)
        return {"stored": True, "content": content}

    async def recall(self, query, session_id="global", top_k=5):
        if self.memory is None:
            return self._error("Memory not available.")
        results = self.memory.query(query=query, session_id=session_id, top_k=top_k)
        return {"query": query, "memories": results}

    async def ide_complete(self, prefix, suffix="", language="python", max_tokens=256, temperature=0.3):
        if suffix:
            prompt = f"<prefix>\n{prefix}\n</prefix>\n<suffix>\n{suffix}\n</suffix>\n<middle>"
        else:
            prompt = f"# Language: {language}\n{prefix}"
        completion = self._generate(prompt, max_tokens, temperature, 0.95)
        return {"completion": completion, "prefix": prefix, "suffix": suffix, "language": language}

    async def ide_explain(self, code, language="python"):
        prompt = f"Explain this {language} code clearly:\n\n```{language}\n{code}\n```\n\nExplanation:"
        explanation = self._generate(prompt, 512, 0.5, 0.95)
        return {"explanation": explanation, "code": code, "language": language}

    async def ide_fix(self, code, error, language="python"):
        prompt = f"Fix this {language} code.\n\nCode:\n```{language}\n{code}\n```\n\nError: {error}\n\nFixed code:"
        fixed = self._generate(prompt, 1024, 0.2, 0.95)
        return {"fixed_code": fixed, "original_code": code, "error": error}

    async def ide_document(self, code, language="python"):
        prompt = f"Write comprehensive docstrings for this {language} code:\n\n```{language}\n{code}\n```\n\nDocumented code:"
        documented = self._generate(prompt, 2048, 0.3, 0.95)
        return {"documented_code": documented, "language": language}

    async def ide_refactor(self, code, instruction, language="python"):
        prompt = f"Refactor this {language} code. Instruction: {instruction}\n\nOriginal:\n```{language}\n{code}\n```\n\nRefactored:"
        refactored = self._generate(prompt, 2048, 0.2, 0.95)
        return {"refactored_code": refactored, "instruction": instruction}

    async def ide_test(self, code, framework="pytest", language="python"):
        prompt = f"Write {framework} tests for this {language} code:\n\n```{language}\n{code}\n```\n\nTests:"
        tests = self._generate(prompt, 2048, 0.3, 0.95)
        return {"tests": tests, "framework": framework, "language": language}

    def _error(self, message):
        return {"error": True, "message": message, "content": message}
