"""
CodeMind Core Engine
Central orchestrator for all AI capabilities
Routes requests to appropriate agents
"""

import torch
import os
from typing import Optional, Dict, Any, List


class CodeMindEngine:
    """
    Master engine that coordinates all CodeMind capabilities.
    Drop-in replacement for OpenAI/Anthropic clients.
    """

    def __init__(
        self,
        model_checkpoint: str = "./checkpoints/checkpoint-best",
        tokenizer_path: str = "./tokenizer/vocab/tokenizer.json",
        device: str = "auto",
        enable_vision: bool = True,
        enable_agents: bool = True,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        self.model = None
        self.tokenizer = None
        self.vision_pipeline = None
        self.agents = {}

        self._load_language_model(model_checkpoint, tokenizer_path)

        if enable_vision:
            self._load_vision()

        if enable_agents:
            self._load_agents()

        print(f"CodeMind Engine ready on {self.device}")

    def _load_language_model(self, checkpoint: str, tokenizer_path: str):
        from model import CodeMindLLM
        from tokenizer import CodeMindTokenizer
        if os.path.exists(checkpoint):
            self.model = CodeMindLLM.from_pretrained(checkpoint).to(self.device).eval()
        else:
            print(f"WARNING: No checkpoint at {checkpoint}. Train first: python train.py")
        if os.path.exists(tokenizer_path):
            self.tokenizer = CodeMindTokenizer(tokenizer_path)

    def _load_vision(self):
        try:
            from vision.image_generator import ImageGenerator
            self.vision_pipeline = ImageGenerator(device=str(self.device))
            print("Vision pipeline loaded")
        except Exception as e:
            print(f"Vision not available: {e}")

    def _load_agents(self):
        from agents.fullstack.fullstack_agent import FullStackAgent
        from agents.gamedev.gamedev_agent import GameDevAgent
        from agents.security.security_agent import SecurityAgent
        from agents.devops.devops_agent import DevOpsAgent
        self.agents = {
            "fullstack": FullStackAgent(self),
            "gamedev": GameDevAgent(self),
            "security": SecurityAgent(self),
            "devops": DevOpsAgent(self),
        }
        print(f"Agents loaded: {list(self.agents.keys())}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Dict[str, Any]:
        """
        OpenAI-compatible chat interface.
        Accepts same format as openai.ChatCompletion.create()
        """
        prompt = self._messages_to_prompt(messages)
        intent = self._detect_intent(prompt)

        if intent in self.agents:
            result = self.agents[intent].handle(prompt)
        else:
            result = self._generate_text(prompt, max_tokens, temperature, top_p)

        return {
            "id": "codemind-response",
            "object": "chat.completion",
            "model": "CodeMind-125M",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(result.split()),
            }
        }

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"Human: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _detect_intent(self, prompt: str) -> Optional[str]:
        p = prompt.lower()
        if any(w in p for w in ["website","frontend","backend","react","vue","django","fastapi","fullstack","html","css","api endpoint"]):
            return "fullstack"
        if any(w in p for w in ["game","pygame","unity","godot","sprite","player","collision","level","score"]):
            return "gamedev"
        if any(w in p for w in ["pentest","ctf","security","vulnerability","nmap","payload","exploit","firewall","audit"]):
            return "security"
        if any(w in p for w in ["docker","kubernetes","ci/cd","deploy","ansible","terraform","pipeline","nginx","systemd"]):
            return "devops"
        return None

    def _generate_text(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        if self.model is None or self.tokenizer is None:
            return "Model not loaded. Run: python train.py"
        from inference.generate import generate_code
        return generate_code(self.model, self.tokenizer, prompt, self.device, max_tokens, temperature, top_p)

    def generate_image(self, prompt: str, width: int = 512, height: int = 512, steps: int = 20) -> str:
        if self.vision_pipeline is None:
            return "Vision not loaded."
        return self.vision_pipeline.generate(prompt, width, height, steps)

    def complete(self, prompt: str, **kwargs) -> str:
        """Simple text completion interface."""
        return self._generate_text(prompt, kwargs.get("max_tokens", 512), kwargs.get("temperature", 0.7), kwargs.get("top_p", 0.95))
