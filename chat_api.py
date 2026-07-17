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
import logging
import os
import sys
import time
import traceback
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


# --- Terminal loglama -------------------------------------------------------
# CHAT_LOG_LEVEL=DEBUG  -> ajanın her adımını (Thought/Action/Observation) bas
# CHAT_LOG_LEVEL=INFO   -> (varsayılan) adımlar + tur özeti
# CHAT_LOG_LEVEL=WARNING-> sadece hatalar
LOG_LEVEL = os.environ.get("CHAT_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,          # uvicorn kendi handler'ını kurduysa ezip bizimkini koy
)
LOG = logging.getLogger("chat")


def _short(text: Any, limit: int = 300) -> str:
    """Uzun metni tek satıra indirip kırp (terminal okunur kalsın)."""
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


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
        from react_agent.tools import TOOLS

        # verbose=True + logger: ajanın her adımı (Thought/Action/Observation)
        # zaman damgalı olarak terminale akar. Susturmak: CHAT_LOG_LEVEL=WARNING
        _agent = ReActAgent(verbose=True, logger=LOG)
        LOG.info("Ajan hazır — model=%s | thinking=%s | %d araç",
                 _agent.llm.model, _agent.llm.enable_thinking, len(TOOLS))
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

    LOG.info("─" * 70)
    LOG.info("[%s] SORU: %s", req.thread_id, _short(req.message, 200))
    t0 = time.time()
    try:
        result = agent.run(req.message, thread_id=req.thread_id)  # aynı thread => bellek
    except Exception as exc:
        # Hatayı yut yerine logla + 500 dön; terminalde tam traceback görünsün.
        LOG.error("[%s] ✗ HATA (%.1fsn): %s: %s",
                  req.thread_id, time.time() - t0, type(exc).__name__, exc)
        LOG.error(traceback.format_exc())
        raise

    # Tur özeti: durum, adım, araçlar, token, süre.
    LOG.info("[%s] %s | adım=%d araç=%d (%s) token=%d (giriş=%d/çıkış=%d) süre=%.1fsn",
             req.thread_id,
             "✓ " + result.status if result.success else "⚠ " + result.status,
             result.steps, result.tool_calls, ", ".join(result.tools_used) or "-",
             result.total_tokens, result.input_tokens, result.output_tokens,
             result.elapsed_seconds)
    LOG.info("[%s] CEVAP: %s", req.thread_id, _short(result.answer, 300))

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
    LOG.info("[%s] bellek temizlendi (silinecek kayıt vardı: %s)", req.thread_id, cleared)
    return {"ok": True, "cleared": cleared, "thread_id": req.thread_id}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "log_file": str(get_logger().path)}


if __name__ == "__main__":
    import uvicorn

    LOG.info("ReAct sohbet sunucusu: http://127.0.0.1:8001")
    LOG.info("Tur kayıtları (JSONL): %s", get_logger().path)
    LOG.info("Log seviyesi: %s  (ayrıntı için CHAT_LOG_LEVEL=DEBUG, sessizlik için WARNING)", LOG_LEVEL)
    # reload=True: kod (araçlar dahil) değişince sunucu kendini yeniler.
    # Bunun çalışması için app'i import string olarak veriyoruz.
    uvicorn.run("chat_api:app", host="127.0.0.1", port=8001, reload=True)
