"""Context Manager — multi-turn conversation with token budget."""

import time
from typing import List, Dict
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    tokens: int = 0


class ContextManager:
    SYSTEM_PROMPT = (
        "You are CodeMind, a powerful standalone AI platform for developers. "
        "You write complete, production-ready code in any programming language. "
        "You never produce skeletons, demos, or TODOs. "
        "You support all frameworks, all languages, and all use cases."
    )

    def __init__(self, max_tokens=8192):
        self.max_tokens = max_tokens
        self.sessions = defaultdict(list)

    def get_context(self, session_id, new_messages):
        for msg in new_messages:
            self.sessions[session_id].append(Message(
                role=msg["role"],
                content=msg["content"],
                tokens=len(msg["content"].split()) * 2,
            ))
        context = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        total = len(self.SYSTEM_PROMPT.split()) * 2
        included = []
        for msg in reversed(self.sessions[session_id]):
            t = msg.tokens or len(msg.content.split()) * 2
            if total + t > self.max_tokens * 0.8:
                break
            total += t
            included.insert(0, {"role": msg.role, "content": msg.content})
        context.extend(included)
        return context

    def update(self, session_id, assistant_response):
        self.sessions[session_id].append(Message(role="assistant", content=assistant_response, tokens=len(assistant_response.split()) * 2))

    def clear(self, session_id):
        self.sessions[session_id] = []
