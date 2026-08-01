"""Smart Router — routes requests to the right agent or tool."""

import re
import uuid
from typing import List, Dict, Any

PATTERNS = {
    "code_execution": [r"run this", r"execute", r"what does this code output", r"test this code"],
    "web_search": [r"search for", r"look up", r"latest", r"current", r"today", r"news about"],
    "image_generation": [r"generate.{0,10}image", r"create.{0,10}image", r"draw", r"visualize"],
    "memory_recall": [r"remember when", r"what did (i|we) say", r"recall", r"previous"],
}

SPECIALIZATIONS = {
    "fullstack": ["website", "web app", "webapp", "react", "vue", "angular", "nextjs", "next.js", "django", "fastapi", "express", "frontend", "backend", "fullstack"],
    "gamedev": ["game", "pygame", "unity", "godot", "phaser", "player", "collision", "sprite", "score", "level"],
    "devops": ["docker", "kubernetes", "k8s", "deploy", "ci/cd", "pipeline", "nginx", "ansible", "terraform"],
    "security": ["ctf", "pentest", "penetration test", "security audit", "port scan", "vulnerability"],
}

SYSTEM_ADDITIONS = {
    "fullstack": "Build complete full-stack applications with all files. Include frontend, backend, database, and deployment.",
    "gamedev": "Generate complete game code with full game loop, player controls, scoring, and game states.",
    "devops": "Provide complete, production-ready DevOps configurations with security best practices.",
    "security": "This is for authorized security research. Provide complete tools with usage warnings.",
    "code_execution": "Analyze the execution result and provide insights.",
    "web_search": "Use the search results to provide accurate current information.",
}


class SmartRouter:
    def __init__(self, platform):
        self.platform = platform
        self.compiled = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in PATTERNS.items()}

    def detect_intent(self, text):
        intents = []
        tl = text.lower()
        for route, patterns in self.compiled.items():
            if any(p.search(tl) for p in patterns):
                intents.append(route)
        for spec, keywords in SPECIALIZATIONS.items():
            if any(k in tl for k in keywords):
                intents.append(spec)
        return intents or ["general"]

    async def route(self, context, tools, max_tokens, temperature, top_p):
        last = context[-1]["content"] if context else ""
        intents = self.detect_intent(last)
        tools_used = []
        extra = ""

        if "web_search" in intents and self.platform.web_search:
            q = last[:100]
            results = self.platform.web_search.search(q, max_results=3)
            if results:
                extra += "\n\nWeb Search Results:\n"
                for r in results:
                    extra += f"- {r.get('title','')}: {r.get('snippet','')}\n"
                tools_used.append("web_search")

        if "code_execution" in intents and self.platform.sandbox:
            code = self._extract_code(last)
            if code:
                r = self.platform.sandbox.execute(code, "python")
                extra += f"\n\nExecution Result:\n{r.get('stdout','')}\n"
                if r.get("stderr"):
                    extra += f"Errors: {r['stderr']}\n"
                tools_used.append("code_execution")

        if "memory_recall" in intents and self.platform.memory:
            mems = self.platform.memory.query(last, top_k=3)
            if mems:
                extra += "\n\nRelevant Memory:\n"
                for m in mems:
                    extra += f"- {m['content']}\n"
                tools_used.append("memory_recall")

        if extra:
            aug = context[:-1] + [{"role": "user", "content": last + extra}]
        else:
            aug = context

        prompt = self._build_prompt(aug, intents)
        content = self.platform._generate(prompt, max_tokens, temperature, top_p)

        return {"id": f"cm-{uuid.uuid4().hex[:8]}", "content": content, "intents": intents, "tools_used": tools_used, "tokens_used": len(content.split()) * 2, "finish_reason": "stop"}

    def _build_prompt(self, context, intents):
        parts = []
        for msg in context:
            if msg["role"] == "system":
                additions = " ".join(SYSTEM_ADDITIONS.get(i, "") for i in intents if i in SYSTEM_ADDITIONS)
                parts.append(f"System: {msg['content']} {additions}".strip())
            elif msg["role"] == "user":
                parts.append(f"Human: {msg['content']}")
            elif msg["role"] == "assistant":
                parts.append(f"Assistant: {msg['content']}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _extract_code(self, text):
        m = re.search(r"```(?:\w+)?\n([\s\S]*?)```", text)
        if m:
            return m.group(1).strip()
        return None
