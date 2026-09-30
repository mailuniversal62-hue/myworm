import os
import time
import json
from collections import defaultdict
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import AsyncGroq

from prompt import SYSTEM_PROMPT

load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
PRIVATE_TOKEN = os.environ["PRIVATE_TOKEN"]
PUBLIC_RATE = int(os.getenv("PUBLIC_RATE_PER_MIN", "5"))
PRIVATE_RATE = int(os.getenv("PRIVATE_RATE_PER_MIN", "60"))
PUBLIC_MAX = int(os.getenv("PUBLIC_MAX_TOKENS", "1024"))
PRIVATE_MAX = int(os.getenv("PRIVATE_MAX_TOKENS", "8192"))

MODEL = "llama-3.1-70b-versatile"
MODEL_FAST = "llama-3.1-8b-instant"

client = AsyncGroq(api_key=GROQ_API_KEY)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- rate limiting (in-memory, per ip) ---
_hits: dict[str, list[float]] = defaultdict(list)

def rate_ok(key: str, limit: int) -> bool:
    now = time.time()
    window = [t for t in _hits[key] if now - t < 60]
    if len(window) >= limit:
        _hits[key] = window
        return False
    window.append(now)
    _hits[key] = window
    return True

def client_ip(req: Request) -> str:
    # cloudflare passes the real ip in this header
    return req.headers.get("cf-connecting-ip") or req.client.host

# --- request shape ---
class ChatRequest(BaseModel):
    messages: list[dict]
    model: str | None = None

# --- streaming helper ---
async def stream_reply(
    messages: list[dict], model: str, max_tokens: int
) -> AsyncGenerator[bytes, None]:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        "stream": True,
        "temperature": 0.8,
        "max_tokens": max_tokens,
    }
    try:
        stream = await client.chat.completions.create(**payload)
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta.encode()
    except Exception as e:
        yield f"\n[upstream error: {e}]".encode()

# ============================================================
# PUBLIC DOOR — anyone on the internet, rate-limited
# ============================================================
@app.post("/public/chat")
async def public_chat(req: Request, body: ChatRequest):
    ip = client_ip(req)
    if not rate_ok(f"pub:{ip}", PUBLIC_RATE):
        raise HTTPException(429, "rate limit — try again in a minute")

    msgs = body.messages[-10:]  # cap context depth
    return StreamingResponse(
        stream_reply(msgs, MODEL_FAST, PUBLIC_MAX),
        media_type="text/plain",
    )

@app.get("/", response_class=HTMLResponse)
async def public_ui():
    return FileResponse("static/index.html")

# ============================================================
# PRIVATE DOOR — requires token, full depth
# ============================================================
@app.post("/private/chat")
async def private_chat(
    req: Request,
    body: ChatRequest,
    authorization: str | None = Header(default=None),
):
    ip = client_ip(req)
    if not rate_ok(f"priv:{ip}", PRIVATE_RATE):
        raise HTTPException(429, "rate limit")

    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
    if token != PRIVATE_TOKEN:
        raise HTTPException(403, "private token required")

    msgs = body.messages  # no context cap
    model = body.model or MODEL
    return StreamingResponse(
        stream_reply(msgs, model, PRIVATE_MAX),
        media_type="text/plain",
    )

@app.get("/private/health")
async def private_health(authorization: str | None = Header(default=None)):
    token = authorization[7:] if authorization and authorization.startswith("Bearer ") else None
    if token != PRIVATE_TOKEN:
        raise HTTPException(403, "no")
    return {"ok": True, "model": MODEL}

@app.get("/health")
async def health():
    return {"ok": True}
