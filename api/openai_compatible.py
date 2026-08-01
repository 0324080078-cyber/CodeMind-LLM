"""
OpenAI-Compatible REST API for CodeMind
Any tool that works with ChatGPT works with CodeMind.
Drop-in replacement for openai Python client.

Usage:
  import openai
  openai.api_base = "http://localhost:8000/v1"
  openai.api_key = "codemind-local"
  # Now use exactly like OpenAI
"""

import time
import uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn


def create_api_app(engine=None) -> FastAPI:
    app = FastAPI(
        title="CodeMind OpenAI-Compatible API",
        description="Drop-in replacement for OpenAI API — runs 100% locally",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state = {"engine": engine}

    # ---- Pydantic Models ----

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = "codemind-125m"
        messages: List[Message]
        max_tokens: int = Field(1024, ge=1, le=8192)
        temperature: float = Field(0.7, ge=0.0, le=2.0)
        top_p: float = Field(0.95, ge=0.0, le=1.0)
        stream: bool = False
        stop: Optional[List[str]] = None

    class CompletionRequest(BaseModel):
        model: str = "codemind-125m"
        prompt: str
        max_tokens: int = Field(512, ge=1, le=4096)
        temperature: float = Field(0.7, ge=0.0, le=2.0)
        top_p: float = Field(0.95, ge=0.0, le=1.0)

    class ImageRequest(BaseModel):
        prompt: str
        n: int = Field(1, ge=1, le=4)
        size: str = "512x512"
        response_format: str = "url"

    class ModelInfo(BaseModel):
        id: str
        object: str = "model"
        created: int
        owned_by: str = "codemind"

    # ---- Routes ----

    @app.get("/")
    async def root():
        return {
            "name": "CodeMind API",
            "version": "1.0.0",
            "openai_compatible": True,
            "endpoints": ["/v1/chat/completions", "/v1/completions", "/v1/images/generations", "/v1/models"],
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                ModelInfo(id="codemind-125m", created=int(time.time())).dict(),
                ModelInfo(id="codemind-vision", created=int(time.time())).dict(),
                ModelInfo(id="gpt-3.5-turbo", created=int(time.time())).dict(),
                ModelInfo(id="gpt-4", created=int(time.time())).dict(),
            ]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        if state["engine"] is None:
            raise HTTPException(503, "Engine not loaded. Run: python codemind_server.py")

        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        response = state["engine"].chat(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response["choices"][0]["message"]["content"],
                },
                "finish_reason": "stop",
            }],
            "usage": response.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        }

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        if state["engine"] is None:
            raise HTTPException(503, "Engine not loaded.")

        result = state["engine"].complete(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "text": result,
                "index": 0,
                "finish_reason": "stop",
            }],
        }

    @app.post("/v1/images/generations")
    async def generate_image(request: ImageRequest):
        if state["engine"] is None:
            raise HTTPException(503, "Engine not loaded.")

        w, h = map(int, request.size.split("x"))
        results = []

        for _ in range(request.n):
            path = state["engine"].generate_image(
                prompt=request.prompt,
                width=w,
                height=h,
            )
            results.append({"url": f"file://{path}", "local_path": path})

        return {
            "created": int(time.time()),
            "data": results,
        }

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "engine_loaded": state["engine"] is not None,
            "openai_compatible": True,
            "version": "1.0.0",
        }

    return app, state
