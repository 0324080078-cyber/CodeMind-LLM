import os, time, torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(title="CodeMind LLM API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
state = {"model":None,"tokenizer":None,"device":None,"loaded":False}

class Req(BaseModel):
    prompt: str
    max_new_tokens: int = Field(256,ge=1,le=1024)
    temperature: float = Field(0.7,ge=0.01,le=2.0)
    top_p: float = Field(0.95,ge=0.0,le=1.0)
    top_k: int = Field(50,ge=0,le=200)

class Resp(BaseModel):
    prompt: str
    generated_text: str
    full_text: str
    time_seconds: float
    model: str = "CodeMind-125M"

@app.on_event("startup")
async def startup():
    from model import CodeMindLLM
    from tokenizer import CodeMindTokenizer
    ckpt = os.environ.get("CODEMIND_CHECKPOINT","./checkpoints/checkpoint-best")
    tok = os.environ.get("CODEMIND_TOKENIZER","./tokenizer/vocab/tokenizer.json")
    if not os.path.exists(ckpt):
        print(f"No checkpoint at {ckpt}. Run: python train.py"); return
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["model"] = CodeMindLLM.from_pretrained(ckpt).to(d).eval()
    state["tokenizer"] = CodeMindTokenizer(tok)
    state["device"] = d
    state["loaded"] = True
    print(f"CodeMind ready on {d}")

@app.get("/health")
async def health():
    return {"status":"ok","loaded":state["loaded"],"device":str(state.get("device"))}

@app.post("/generate", response_model=Resp)
async def generate(req: Req):
    if not state["loaded"]:
        raise HTTPException(503,"Model not loaded. Run: python train.py")
    from inference.generate import generate_code
    t0 = time.time()
    g = generate_code(state["model"],state["tokenizer"],req.prompt,state["device"],req.max_new_tokens,req.temperature,req.top_p,req.top_k)
    return Resp(prompt=req.prompt,generated_text=g,full_text=req.prompt+g,time_seconds=round(time.time()-t0,3))

if __name__ == "__main__":
    uvicorn.run("inference.server:app",host="0.0.0.0",port=8000,reload=False)
