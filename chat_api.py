"""ReAct ajanını bir web UI'da elle test etmek için FastAPI sohbet sunucusu.

- Arayüzü (index.html) servis eder.
- /api/chat: gelen mesajı ajana verir (thread_id ile çok-turlu / bellekli),
  metrikleri + adım adım trace'i döndürür.
- /api/reset: bir thread'in belleğini siler.
- Her tur ayrıntısıyla scratch/chat_logs/chat_<zaman>.jsonl dosyasına yazılır.

Çalıştırma:
    python chat_api.py
    -> http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# .env varsa yükle (HF_TOKEN, HF_MODEL). python-dotenv opsiyonel.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parent
INDEX_HTML = ROOT_DIR / "index.html"
LOG_DIR = ROOT_DIR / "scratch" / "chat_logs"


# --- İstek/yanıt modelleri --------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    thread_id: str


class ResetRequest(BaseModel):
    thread_id: str


class TraceStepOut(BaseModel):
    adim: int
    thought: str | None = None
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None


class ChatResponse(BaseModel):
    answer: str | None
    success: bool
    steps: int
    tool_calls: int
    tools_used: list[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    duration_ms: int
    trace: list[TraceStepOut]


# --- Sohbet log'u -----------------------------------------------------------


class ChatLogger:
    """Her sohbet turunu tek satır JSON (JSONL) biçiminde diske yazar."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = log_dir / f"chat_{stamp}.jsonl"
        self._turns: dict[str, int] = defaultdict(int)

    def next_turn(self, thread_id: str) -> int:
        self._turns[thread_id] += 1
        return self._turns[thread_id]

    def write(self, obj: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


# --- Ajan (tekil örnek; bellek thread_id başına ajanın içinde tutulur) ------

_agent = None
_logger: ChatLogger | None = None


def get_agent():
    global _agent
    if _agent is None:
        from react_agent import ReActAgent

        # verbose=False: sunucu konsolunu şişirmesin. Model .env'den (HF_MODEL).
        _agent = ReActAgent(verbose=False)
    return _agent


def get_logger() -> ChatLogger:
    global _logger
    if _logger is None:
        _logger = ChatLogger(LOG_DIR)
    return _logger


def _serialize_trace(trace: list) -> list[dict[str, Any]]:
    return [
        {
            "adim": i,
            "thought": step.thought,
            "action": step.action,
            "action_input": step.action_input,
            "observation": step.observation,
        }
        for i, step in enumerate(trace, 1)
    ]


# --- FastAPI uygulaması ------------------------------------------------------

app = FastAPI(title="ReAct Ajan Sohbet UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # def (async değil): agent.run senkron; FastAPI bunu threadpool'da çalıştırır,
    # böylece uzun bir LLM çağrısı event loop'u bloklamaz.
    agent = get_agent()
    result = agent.run(req.message, thread_id=req.thread_id)  # aynı thread => bellek

    payload = {
        "answer": result.answer,
        "success": result.success,
        "steps": result.steps,
        "tool_calls": result.tool_calls,
        "tools_used": result.tools_used,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
        "duration_ms": int(result.elapsed_seconds * 1000),
        "trace": _serialize_trace(result.trace),
    }

    # Diske log yaz (turun tam kaydı).
    logger = get_logger()
    turn = logger.next_turn(req.thread_id)
    logger.write({"thread_id": req.thread_id, "turn": turn, "message": req.message, **payload})

    return ChatResponse(**payload)


@app.post("/api/reset")
def reset(req: ResetRequest) -> dict[str, Any]:
    """Bir thread'in sunucudaki belleğini siler (Temizle butonu bunu çağırır)."""
    cleared = get_agent().reset_memory(req.thread_id)
    return {"ok": True, "cleared": cleared, "thread_id": req.thread_id}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "log_file": str(get_logger().path)}


if __name__ == "__main__":
    import uvicorn

    print("ReAct sohbet sunucusu: http://127.0.0.1:8000")
    # reload=True: kod (araçlar dahil) değişince sunucu kendini yeniler.
    # Bunun çalışması için app'i import string olarak veriyoruz.
    uvicorn.run("chat_api:app", host="127.0.0.1", port=8001, reload=True)
