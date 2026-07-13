

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .llm import HuggingFaceLLM
from .prompts import build_system_prompt, build_user_prompt
from .tools import render_tool_descriptions, run_tool, TOOLS

# Modelin çıktısından alanları çeken düzenli ifadeler (regex).
THOUGHT_RE = re.compile(r"Thought\s*:\s*(.*?)(?:\n(?:Action|Final Answer)|$)", re.IGNORECASE | re.DOTALL)
ACTION_RE = re.compile(r"Action\s*:\s*(.*?)\n", re.IGNORECASE)
ACTION_INPUT_RE = re.compile(r"Action Input\s*:\s*(.*?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
FINAL_RE = re.compile(r"Final Answer\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
# Final Answer'dan sonra model yine ReAct döngüsüne devam ederse, cevabı bu
# işaretlerin ilkinde keseriz (emniyet kelepçesi).
# DİKKAT: yalnızca ReAct-yapısal işaretleri kesiyoruz. Markdown başlık/kalın/liste
# KESİLMEZ — meşru raporlar (## Bölüm, **Başlık:**, 1. madde) bunları kullanıyor.
# Reasoning sızıntısını asıl önleyen şey system/user rol ayrımı + yalnızca
# `content` okumaktır (Qwen3 düşünmeyi ayrı `reasoning_content` alanında verir).
ANSWER_CUT_RE = re.compile(
    r"\n\s*(?:"
    r"Thought\s*:|Action\s*:|Action Input\s*:|Observation\s*:|"  # ReAct alanları
    r"Question\s*:|Final Answer\s*:|"                            # ikinci soru/cevap sızıntısı
    r"Final Review\b|Wait[,:]"                                    # güçlü İngilizce düşünme sinyali
    r")",
    re.IGNORECASE,
)


@dataclass
class TraceStep:
    """Döngünün tek bir adımı — UI'da 'adımları göster' altında listelenir.
    Ayrıca eval şeması (a.json) için adım bazında metrik/zaman damgası tutar."""
    thought: str | None = None
    reasoning: str | None = None        # modelin düşünme çıktısı (reasoning_content)
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None
    # --- metrikler (eval şeması için) ---
    started_at: str | None = None       # ISO-8601
    ended_at: str | None = None         # ISO-8601
    input_tokens: int = 0               # bu adımdaki LLM çağrısı
    output_tokens: int = 0
    llm_ms: float = 0.0                 # LLM inference süresi
    tool_ms: float = 0.0                # araç yürütme süresi (varsa)
    tool_success: bool | None = None    # araç çağrısı başarılı mı (araç yoksa None)


@dataclass
class RunResult:
    """Bir turun tüm sonucu + metrikleri. API bu nesneyi JSON'a çevirir."""
    answer: str | None = None
    success: bool = False
    steps: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0.0
    started_at: str | None = None       # ISO-8601 (turun başı)
    ended_at: str | None = None         # ISO-8601 (turun sonu)
    status: str = "success"             # success | partial | timeout | error
    scratchpad: str = ""                # modele beslenen tam çalışma izi (transcript)
    trace: list = field(default_factory=list)

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    @property
    def tools_used(self):
        return [s.action for s in self.trace if s.action]


class ReActAgent:
    def __init__(self, llm=None, max_steps=12, verbose=True):
        self.llm = llm or HuggingFaceLLM()
        self.max_steps = max_steps      # sonsuz döngüye karşı güvenlik sınırı
        self.verbose = verbose
        # thread_id -> [(soru, cevap), ...]  : çok-turlu bellek
        self._memory = {}

    def _log(self, text):
        if self.verbose:
            print(text)

    def reset_memory(self, thread_id):
        """Bir thread'in belleğini siler. Silinecek bir şey varsa True döner."""
        return self._memory.pop(thread_id, None) is not None

    def run(self, question, thread_id=None):
        start = time.time()
        result = RunResult(started_at=self._now_iso())

        tools_text = render_tool_descriptions()
        tool_names = ", ".join(TOOLS.keys())
        history = self._memory.get(thread_id, []) if thread_id else []

        # Talimatlar sabit (system rolü). Çalışma izini (scratchpad) her adımda
        # büyütüp user mesajına ekliyoruz — turun hafızası budur.
        system_prompt = build_system_prompt(tools_text, tool_names)
        scratchpad = ""

        for step in range(1, self.max_steps + 1):
            self._log(f"\n===== Adım {step} =====")
            result.steps = step
            step_started = self._now_iso()

            user_prompt = build_user_prompt(question, history, scratchpad)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            # Modeli "Observation:" görünce durdur; gözlemi kendisi uydurmasın.
            t_llm = time.time()
            resp = self.llm.chat(messages, stop=["Observation:"])
            llm_ms = (time.time() - t_llm) * 1000
            result.input_tokens += resp.input_tokens
            result.output_tokens += resp.output_tokens

            output = resp.text.strip()
            self._log(output)

            thought = self._extract(THOUGHT_RE, output)
            # Bu adımın ortak metrikleri (araçlı/araçsız her dalda kullanılır).
            step_meta = dict(
                thought=thought, reasoning=resp.reasoning or None,
                started_at=step_started,
                input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                llm_ms=llm_ms,
            )

            # 1) Model nihai cevabı verdi mi?
            final = FINAL_RE.search(output)
            if final:
                result.answer = self._clean_final(final.group(1))
                result.success = True
                result.status = "success"
                scratchpad += output   # final adımı da transcript'e ekle
                result.trace.append(TraceStep(ended_at=self._now_iso(), **step_meta))
                break

            # 2) Bir eylem (Action) istedi mi?
            raw_action = self._extract(ACTION_RE, output)
            raw_input = self._extract(ACTION_INPUT_RE, output) or ""
            action, action_input = self._resolve_action(raw_action, raw_input, output)

            if not action:
                # Ne Final Answer ne Action var — modeli formata geri çağır.
                nudge = "Format hatalı. Lütfen 'Action:' veya 'Final Answer:' kullan."
                result.trace.append(TraceStep(observation=nudge, ended_at=self._now_iso(), **step_meta))
                scratchpad += output + f"\nObservation: {nudge}\n"
                continue

            # Aracı BİZ çalıştırırız — gerçek "eylem" burada gerçekleşir.
            t_tool = time.time()
            observation = run_tool(action, action_input)
            tool_ms = (time.time() - t_tool) * 1000
            result.tool_calls += 1
            self._log(f"Observation: {observation}")

            result.trace.append(TraceStep(
                action=action, action_input=action_input, observation=observation,
                tool_ms=tool_ms, tool_success=self._tool_ok(observation),
                ended_at=self._now_iso(), **step_meta,
            ))
            # Modelin ürettiği metni + gerçek gözlemi scratchpad'e ekleyip sürdür.
            scratchpad += output + f"\nObservation: {observation}\n"
        else:
            # for döngüsü break'siz bitti = adım sınırına ulaşıldı.
            result.answer = "Adım sınırına ulaşıldı, kesin bir cevap üretilemedi."
            result.status = "partial"

        result.elapsed_seconds = time.time() - start
        result.ended_at = self._now_iso()
        result.scratchpad = scratchpad

        # Turu belleğe yaz (sadece başarılı cevapları hatırlamak yeterli).
        if thread_id and result.answer:
            self._memory.setdefault(thread_id, []).append((question, result.answer))

        return result

    @staticmethod
    def _now_iso():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _tool_ok(observation):
        """Araç çıktısına bakıp çağrının başarılı olup olmadığını kestir.
        Araçlarımız hata durumunda tanınabilir metinler döndürüyor."""
        obs = (observation or "")[:80]
        markers = ("HATA:", "hatası:", "kurulu değil", "Ağ hatası", "ayarlı değil")
        return not any(m in obs for m in markers)

    @staticmethod
    def _extract(regex, text):
        m = regex.search(text)
        return m.group(1).strip() if m else None

    @staticmethod
    def _clean(text):
        """Markdown/backtick/tırnak süslerini temizle (model 'Action: **tool**'
        ya da '`tool`' yazabiliyor)."""
        return (text or "").strip().strip("*`\"' ").strip()

    @staticmethod
    def _clean_final(text):
        """Final Answer'ı temizle: model cevaptan sonra yine akıl yürütmeye
        devam ederse (Thought/Final Review/Wait...), cevabı orada keseriz."""
        answer = (text or "").strip()
        cut = ANSWER_CUT_RE.search(answer)
        if cut:
            answer = answer[: cut.start()].strip()
        return answer

    def _resolve_action(self, raw_action, raw_input, output):
        """Model çıktısındaki araç adını temizleyip bilinen araçlara eşler.
        - Markdown süslerini atar.
        - 'tool(girdi)' biçimini ayırır.
        - Ad birebir tutmuyorsa, çıktıda geçen bilinen bir araç adını arar.
        """
        action = self._clean(raw_action)
        action_input = self._clean(raw_input)

        # 'get_company_news(NVDA)' gibi parantezli biçim
        if "(" in action and action.endswith(")"):
            name, _, inside = action.partition("(")
            action = self._clean(name)
            if not action_input:
                action_input = self._clean(inside[:-1])

        if action in TOOLS:
            return action, action_input

        # Birebir tutmadı: temizlenmiş adın içinde ya da tüm çıktıda geçen
        # bilinen bir araç adı var mı? (markdown gürültüsüne karşı son çare)
        for name in TOOLS:
            if name in action or name in output:
                return name, action_input
        return action, action_input
