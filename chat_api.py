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
import re
import sys
import threading
import time
import traceback
from collections import defaultdict
from contextvars import ContextVar
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
# Oturum (thread) başına insan-okur log dosyası. Plan-Execute tarafındaki
# logs/<thread_id>.log ile AYNI düzen — iki mimarinin akışı yan yana okunabilsin.
SESSION_LOG_DIR = Path(os.getenv("LOG_DIR") or (ROOT_DIR / "logs"))


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


# --- Oturum (thread) log dosyaları ------------------------------------------
# Plan-Execute tarafıyla simetrik: her sohbet kendi logs/<thread_id>.log dosyasına
# yazar. Terminalde yalnızca kısa özet kalır; adım adım TAM akış (her ReAct adımı:
# Thought / Action / Observation) dosyaya iner. Paralel sohbetler karışmaz, koşu
# bittikten sonra da incelenebilir.
#
# ReAct ajanı TEKİL ve istekleri FastAPI threadpool'da koşuyor; ajanın tek bir
# `logger` alanı var. İstek başına farklı dosyaya yazmak için logger'ı ContextVar
# ile enjekte ediyoruz: _RequestLogger, aktif isteğin oturum dosyasına yönlendirir.
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")
_session_loggers: dict[str, logging.Logger] = {}
_session_lock = threading.Lock()
# O an işlenen isteğin oturum logger'ı (yoksa None → sadece terminal).
_active_session: ContextVar[logging.Logger | None] = ContextVar("_active_session", default=None)


def _session_logger(thread_id: str) -> logging.Logger:
    """thread_id'ye özgü dosya logger'ı — ilk çağrıda kurulur, sonra önbellekten.

    thread_id istek gövdesinden gelir (dışarıdan kontrol edilir); dosya adına
    girmeden önce ayraç/nokta içermeyen bir alt kümeye indirilir — aksi halde
    '../x' gibi bir değer log dizininin dışına yazabilirdi.
    """
    safe = _UNSAFE_NAME.sub("_", str(thread_id or ""))[:64] or "anon"
    with _session_lock:
        logger = _session_loggers.get(safe)
        if logger is not None:
            return logger
        SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"react.session.{safe}")
        logger.setLevel(logging.INFO)
        logger.propagate = False  # oturum ayrıntısı terminale TEKRAR basılmasın
        handler = logging.FileHandler(SESSION_LOG_DIR / f"{safe}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.info("=" * 72)
        logger.info("OTURUM %s · başladı %s", safe, time.strftime("%Y-%m-%d %H:%M:%S"))
        agent = get_agent()
        logger.info("mimari=ReAct · model=%s · thinking=%s",
                    agent.llm.model, agent.llm.enable_thinking)
        logger.info("(bu dosya canlı güncellenir — `tail -f` ile izleyebilirsin)")
        logger.info("=" * 72)
        _session_loggers[safe] = logger
        return logger


class _RequestLogger:
    """ReActAgent'a verilen 'logger'. Ajanın _log çağrılarını hem terminale
    (kısa özet için değil — tam akış terminalde de kalsın istenirse) hem de
    o anki isteğin oturum dosyasına yönlendirir.

    Ajan yalnızca .info(str) çağırıyor; bu sınıf o tek metodu karşılar."""

    def info(self, msg, *args, **kwargs) -> None:
        text = msg % args if args else msg
        session = _active_session.get()
        target = session if session is not None else LOG
        # Ajan modelin ham çıktısını (Thought/Action/Action Input) tek blok olarak
        # basıyor. Çok satırlıysa satır satır yaz ki her satır kendi zaman damgasını
        # alsın ve hizalama bozulmasın; devam satırlarını hafifçe girintile.
        lines = str(text).splitlines() or [""]
        for i, line in enumerate(lines):
            target.info("%s", line if i == 0 else f"    {line}")


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

        # verbose=True + _RequestLogger: ajanın her adımı (Thought/Action/
        # Observation) o anki isteğin oturum dosyasına akar (bkz. _session_logger).
        # İstek bağlamı dışındaysa terminale düşer. Susturmak: CHAT_LOG_LEVEL=WARNING
        _agent = ReActAgent(verbose=True, logger=_RequestLogger())
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
    slog = _session_logger(req.thread_id)

    # Terminalde yalnızca kısa 'başladı' satırı; tam akış oturum dosyasına.
    LOG.info("▶ [%s] SORU: %s", req.thread_id, _short(req.message, 140))
    slog.info("")
    slog.info("─" * 72)
    slog.info("▶ SORU: %s", _short(req.message, 500))
    slog.info("─" * 72)

    t0 = time.time()
    token = _active_session.set(slog)  # ajanın _log çağrılarını bu dosyaya yönlendir
    try:
        result = agent.run(req.message, thread_id=req.thread_id)  # aynı thread => bellek
    except Exception as exc:
        # Hatayı logla (tam traceback dosyaya) ama 500 yerine GEÇERLİ bir JSON
        # cevabı dön — aksi halde UI düz metin 500'ü parse edemeyip çöküyor.
        dt = time.time() - t0
        LOG.error("[%s] ✗ HATA (%.1fsn): %s: %s", req.thread_id, dt, type(exc).__name__, exc)
        slog.error("✗ HATA (%.1fsn): %s: %s", dt, type(exc).__name__, exc)
        slog.error(traceback.format_exc())
        # Kullanıcıya anlaşılır mesaj (özellikle "model meşgul" / 429 durumunda).
        msg = str(exc)
        if "429" in msg or "engine_overloaded" in msg or "busy" in msg.lower():
            friendly = "Model şu an meşgul (sağlayıcı aşırı yüklü). Lütfen birkaç saniye sonra tekrar dene."
        elif "timeout" in msg.lower() or "timed out" in msg.lower():
            friendly = "Model zaman aşımına uğradı. Tekrar dener misin?"
        else:
            friendly = f"Bir hata oluştu: {msg[:200]}"
        return ChatResponse(
            answer=friendly, success=False, steps=0, tool_calls=0, tools_used=[],
            input_tokens=0, output_tokens=0, total_tokens=0,
            duration_ms=int(dt * 1000), trace=[],
        )
    finally:
        _active_session.reset(token)

    # Tur özeti: durum, adım, araçlar, token, süre. Terminale tek satır, dosyaya
    # cevabıyla birlikte.
    ozet = ("%s | adım=%d araç=%d (%s) token=%d (giriş=%d/çıkış=%d) süre=%.1fsn" % (
        "✓ " + result.status if result.success else "⚠ " + result.status,
        result.steps, result.tool_calls, ", ".join(result.tools_used) or "-",
        result.total_tokens, result.input_tokens, result.output_tokens,
        result.elapsed_seconds))
    LOG.info("[%s] %s", req.thread_id, ozet)
    slog.info("%s", ozet)
    slog.info("CEVAP: %s", _short(result.answer, 2000))

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
    _session_logger(req.thread_id).info(
        "🗑 BELLEK SIFIRLANDI (silinecek geçmiş var mıydı: %s)", cleared)
    return {"ok": True, "cleared": cleared, "thread_id": req.thread_id}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "log_file": str(get_logger().path)}


if __name__ == "__main__":
    import uvicorn

    LOG.info("ReAct sohbet sunucusu: http://127.0.0.1:8001")
    LOG.info("Tur kayıtları (JSONL): %s", get_logger().path)
    LOG.info("Oturum log'ları: %s  (her sohbet kendi <thread_id>.log dosyasına yazar)", SESSION_LOG_DIR)
    LOG.info("Log seviyesi: %s  (ayrıntı için CHAT_LOG_LEVEL=DEBUG, sessizlik için WARNING)", LOG_LEVEL)
    # reload=True: kod (araçlar dahil) değişince sunucu kendini yeniler.
    # Bunun çalışması için app'i import string olarak veriyoruz.
    uvicorn.run("chat_api:app", host="127.0.0.1", port=8001, reload=True)
