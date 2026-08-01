"""
CodeMind REST API — Full OpenAI-compatible + extensions
"""

import time
import uuid
import json
import hashlib
import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


def load_api_keys():
    p = "./api_keys/keys.json"
    if not os.path.exists(p):
        return set()
    with open(p) as f:
        return {k["key_hash"] for k in json.load(f)}


def verify_api_key(authorization: str = Header(None)):
    if os.environ.get("CODEMIND_API_KEY_REQUIRED", "true").lower() != "true":
        return "anonymous"
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "API key required. Header: Authorization: Bearer cm-xxx")
    key = authorization.replace("Bearer ", "").strip()
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    valid = load_api_keys()
    if valid and key_hash not in valid:
        raise HTTPException(401, "Invalid API key. Run: python codemind.py generate-key")
    return key


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "codemind-v2"
    messages: List[ChatMessage]
    max_tokens: int = Field(2048, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    stream: bool = False
    session_id: Optional[str] = None
    tools: Optional[List[str]] = None

class CompletionRequest(BaseModel):
    model: str = "codemind-v2"
    prompt: str
    max_tokens: int = Field(1024, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    language: Optional[str] = None

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, bad quality, distorted"
    n: int = Field(1, ge=1, le=4)
    size: str = "512x512"
    steps: int = Field(20, ge=5, le=50)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)
    seed: Optional[int] = None

class AudioRequest(BaseModel):
    audio_path: str
    language: Optional[str] = None

class SpeechRequest(BaseModel):
    input: str
    output_path: Optional[str] = None

class ExecuteRequest(BaseModel):
    code: str
    language: str = "python"
    stdin: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    max_results: int = Field(5, ge=1, le=20)

class MemoryStoreRequest(BaseModel):
    content: str
    session_id: str = "global"

class MemoryRecallRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)

class IDECompleteRequest(BaseModel):
    prefix: str
    suffix: str = ""
    language: str = "python"
    max_tokens: int = Field(256, ge=1, le=1024)
    temperature: float = Field(0.3, ge=0.0, le=1.0)

class IDERequest(BaseModel):
    code: str
    language: str = "python"

class IDEFixRequest(BaseModel):
    code: str
    error: str
    language: str = "python"

class IDERefactorRequest(BaseModel):
    code: str
    instruction: str
    language: str = "python"

class IDETestRequest(BaseModel):
    code: str
    framework: str = "pytest"
    language: str = "python"


def create_app(platform) -> FastAPI:
    app = FastAPI(
        title="CodeMind AI Platform API",
        description="Standalone AI Platform. OpenAI-compatible. IDE-ready. Developer-first.",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    state = {"platform": platform}

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return """<!DOCTYPE html>
<html><head><title>CodeMind AI</title>
<style>body{background:#0d1117;color:#e6edf3;font-family:monospace;padding:40px}
h1{color:#58a6ff;font-size:2.2em}
.badge{background:#21262d;padding:5px 12px;border-radius:16px;margin:3px;display:inline-block;color:#79c0ff;border:1px solid #30363d;font-size:.85em}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:18px;margin:14px 0}
code,pre{background:#21262d;padding:2px 6px;border-radius:4px;color:#79c0ff}
pre{padding:14px;overflow-x:auto;display:block}a{color:#58a6ff}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}</style></head>
<body>
<h1>&#128081; CodeMind AI Platform v2.0</h1>
<p>Your lifetime standalone AI. No external AI APIs. Everything runs locally on your hardware.</p>
<div style="margin:16px 0">
<span class="badge">&#10003; Standalone</span>
<span class="badge">&#10003; OpenAI Compatible</span>
<span class="badge">&#10003; IDE Ready</span>
<span class="badge">&#10003; Image Generation</span>
<span class="badge">&#10003; Voice I/O</span>
<span class="badge">&#10003; Code Execution</span>
<span class="badge">&#10003; Vector Memory</span>
<span class="badge">&#10003; Web Search</span>
</div>
<div class="grid">
<div class="card"><h3>API Docs</h3><a href="/docs">Swagger UI</a> | <a href="/redoc">ReDoc</a></div>
<div class="card"><h3>Generate API Key</h3><code>python codemind.py generate-key</code></div>
</div>
<div class="card"><h3>Quick Start — Python</h3>
<pre>from sdk.python import CodeMindClient
client = CodeMindClient(api_key="your-key")
print(client.chat("Build a FastAPI app"))
img = client.generate_image("cyberpunk city")
result = client.execute("print('hello world')")</pre></div>
<div class="card"><h3>OpenAI Drop-in</h3>
<pre>import openai
openai.api_key = "your-codemind-key"
openai.base_url = "http://localhost:8000/v1"
# Works exactly like OpenAI</pre></div>
</body></html>"""

    @app.get("/v1/models")
    async def models(key: str = Depends(verify_api_key)):
        return {"object": "list", "data": [
            {"id": "codemind-v2", "object": "model", "created": int(time.time()), "owned_by": "codemind"},
            {"id": "codemind-code", "object": "model", "created": int(time.time()), "owned_by": "codemind"},
            {"id": "gpt-3.5-turbo", "object": "model", "created": int(time.time()), "owned_by": "codemind"},
            {"id": "gpt-4", "object": "model", "created": int(time.time()), "owned_by": "codemind"},
        ]}

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest, key: str = Depends(verify_api_key)):
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        sid = req.session_id or f"s-{uuid.uuid4().hex[:8]}"
        r = await state["platform"].chat(messages=msgs, session_id=sid, max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p, tools=req.tools)
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion", "created": int(time.time()), "model": req.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": r["content"]}, "finish_reason": "stop"}], "usage": {"prompt_tokens": sum(len(m["content"].split()) for m in msgs), "completion_tokens": len(r["content"].split()), "total_tokens": sum(len(m["content"].split()) for m in msgs) + len(r["content"].split())}, "codemind": {"tools_used": r.get("tools_used", []), "session_id": sid}}

    @app.post("/v1/completions")
    async def complete(req: CompletionRequest, key: str = Depends(verify_api_key)):
        r = await state["platform"].complete(prompt=req.prompt, max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p, language=req.language)
        return {"id": f"cmpl-{uuid.uuid4().hex[:8]}", "object": "text_completion", "created": int(time.time()), "model": req.model, "choices": [{"text": r["completion"], "index": 0, "finish_reason": "stop"}]}

    @app.post("/v1/images/generations")
    async def images(req: ImageRequest, key: str = Depends(verify_api_key)):
        w, h = map(int, req.size.split("x"))
        results = []
        for i in range(req.n):
            seed = (req.seed + i) if req.seed else None
            r = await state["platform"].generate_image(prompt=req.prompt, negative_prompt=req.negative_prompt, width=w, height=h, steps=req.steps, guidance_scale=req.guidance_scale, seed=seed)
            path = r.get("image_path", "")
            results.append({"url": f"/outputs/images/{os.path.basename(path)}", "local_path": path})
        return {"created": int(time.time()), "data": results}

    @app.post("/v1/audio/transcriptions")
    async def transcribe(req: AudioRequest, key: str = Depends(verify_api_key)):
        r = await state["platform"].transcribe_audio(req.audio_path, req.language)
        return {"text": r.get("text", "")}

    @app.post("/v1/audio/speech")
    async def tts(req: SpeechRequest, key: str = Depends(verify_api_key)):
        r = await state["platform"].speak(req.input, req.output_path)
        return {"audio_path": r.get("audio_path", "")}

    @app.post("/v1/execute")
    async def execute(req: ExecuteRequest, key: str = Depends(verify_api_key)):
        return await state["platform"].execute_code(req.code, req.language, req.stdin)

    @app.post("/v1/search")
    async def search(req: SearchRequest, key: str = Depends(verify_api_key)):
        return await state["platform"].search_web(req.query, req.max_results)

    @app.post("/v1/memory/store")
    async def mem_store(req: MemoryStoreRequest, key: str = Depends(verify_api_key)):
        return await state["platform"].remember(req.content, req.session_id)

    @app.post("/v1/memory/recall")
    async def mem_recall(req: MemoryRecallRequest, key: str = Depends(verify_api_key)):
        return await state["platform"].recall(req.query, req.session_id or "global", req.top_k)

    @app.get("/v1/memory/stats")
    async def mem_stats(key: str = Depends(verify_api_key)):
        mem = state["platform"].memory
        return {"total_items": mem.count() if mem else 0}

    @app.post("/v1/ide/complete")
    async def ide_complete(req: IDECompleteRequest, key: str = Depends(verify_api_key)):
        return await state["platform"].ide_complete(req.prefix, req.suffix, req.language, req.max_tokens, req.temperature)

    @app.post("/v1/ide/explain")
    async def ide_explain(req: IDERequest, key: str = Depends(verify_api_key)):
        return await state["platform"].ide_explain(req.code, req.language)

    @app.post("/v1/ide/fix")
    async def ide_fix(req: IDEFixRequest, key: str = Depends(verify_api_key)):
        return await state["platform"].ide_fix(req.code, req.error, req.language)

    @app.post("/v1/ide/document")
    async def ide_doc(req: IDERequest, key: str = Depends(verify_api_key)):
        return await state["platform"].ide_document(req.code, req.language)

    @app.post("/v1/ide/refactor")
    async def ide_refactor(req: IDERefactorRequest, key: str = Depends(verify_api_key)):
        return await state["platform"].ide_refactor(req.code, req.instruction, req.language)

    @app.post("/v1/ide/test")
    async def ide_test(req: IDETestRequest, key: str = Depends(verify_api_key)):
        return await state["platform"].ide_test(req.code, req.framework, req.language)

    @app.websocket("/v1/ws/stream")
    async def ws_stream(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                r = await state["platform"].chat(messages=data.get("messages", []), session_id=data.get("session_id", "ws"), max_tokens=data.get("max_tokens", 1024), temperature=data.get("temperature", 0.7))
                content = r["content"]
                words = content.split()
                import asyncio
                for i, word in enumerate(words):
                    await websocket.send_json({"type": "token", "content": word + (" " if i < len(words)-1 else ""), "done": False})
                    await asyncio.sleep(0.02)
                await websocket.send_json({"type": "done", "content": "", "done": True})
        except WebSocketDisconnect:
            pass

    @app.get("/health")
    async def health():
        p = state["platform"]
        return {"status": "ok", "version": "2.0.0", "llm_loaded": p.llm is not None, "vision_loaded": p.vision is not None, "audio_loaded": p.audio_stt is not None, "memory_loaded": p.memory is not None, "sandbox_loaded": p.sandbox is not None, "web_search_loaded": p.web_search is not None, "device": str(p.device), "timestamp": int(time.time())}

    return app
