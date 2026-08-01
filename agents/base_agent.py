"""
Base Agent class all specialized agents inherit from
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseAgent(ABC):
    """Base class for all CodeMind agents."""

    def __init__(self, engine):
        self.engine = engine

    @abstractmethod
    def handle(self, prompt: str) -> str:
        """Handle a prompt and return the response."""
        pass

    def _generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
        """Generate text using the core LLM."""
        return self.engine._generate_text(prompt, max_tokens, temperature, 0.95)

    def _build_system_prompt(self, role: str, instructions: str) -> str:
        return f"You are {role}.\n{instructions}\n\nRespond with complete, working, production-ready code only."
