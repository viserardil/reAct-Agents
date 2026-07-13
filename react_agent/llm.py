
import os
import time
from dataclasses import dataclass

import requests

# SAĞLAYICI-BAĞIMSIZ istemci: OpenAI-uyumlu /chat/completions konuşan HERHANGİ bir
# sağlayıcıyla çalışır (HF Router, OpenAI, Gemini-compat, Groq, Together, OpenRouter,
# DeepInfra, yerel Ollama/vLLM...). Base URL, anahtar ve model env'den okunur;
# LLM_* ayarlıysa HF_*'ın yerine geçer, hiçbiri yoksa HF Router'a düşer.
DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-122B-A10B:deepinfra"


def _first(*names):
    """Verilen env adlarından ilk dolu olanı döndürür."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


class LLMError(Exception):
    """LLM çağrısı başarısız olduğunda fırlatılır."""


@dataclass
class LLMResponse:
    """Bir LLM çağrısının sonucu: üretilen metin + token sayaçları.
    Token'ları API'nin `usage` alanından okuyoruz; UI'da maliyeti göstermek için."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning: str = ""   # modelin düşünme çıktısı (Qwen3 vb. `reasoning_content`)


class HuggingFaceLLM:
    """Adı geriye dönük uyum için 'HuggingFaceLLM'; artık sağlayıcı-bağımsızdır."""

    def __init__(self, model=None, api_key=None, base_url=None, temperature=0.0,
                 timeout=None, max_tokens=4096, max_retries=None):
        # Anahtar: parametre > LLM_API_KEY > HF_TOKEN > yaygın sağlayıcı adları.
        self.api_key = api_key or _first(
            "LLM_API_KEY", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
            "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY", "GROQ_API_KEY",
        )
        if not self.api_key:
            raise LLMError(
                "LLM API anahtarı bulunamadı. .env'e LLM_API_KEY=<anahtar> ekle "
                "(HuggingFace kullanıyorsan HF_TOKEN da olur)."
            )
        self.model = model or _first("LLM_MODEL", "HF_MODEL") or DEFAULT_MODEL

        # Base URL (LLM_BASE_URL > HF_BASE_URL > varsayılan). Uçtaki /chat/completions
        # yoksa ekleriz; sağlayıcılar base'i genelde .../v1 diye verir.
        base = (base_url or _first("LLM_BASE_URL", "HF_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.url = base if base.endswith("/chat/completions") else base + "/chat/completions"

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout if timeout is not None else float(os.environ.get("LLM_TIMEOUT", "90"))
        self.max_retries = max_retries if max_retries is not None else int(os.environ.get("LLM_MAX_RETRIES", "2"))

    def chat(self, messages, stop=None):
        """
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        stop:     durdurma dizileri listesi. Model bu metni üretmeye başlayınca
                  jenerasyonu keser. ReAct'te "Observation:" ile durdururuz ki
                  model gözlemi kendisi uydurmasın.
        Dönüş:    LLMResponse (metin + token sayaçları).

        Sağlayıcı bir çağrıyı askıya alırsa timeout (varsayılan 90s) devreye girer;
        ağ hatası / 429 / 5xx durumunda birkaç kez (LLM_MAX_RETRIES) yeniden dener.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stop:
            payload["stop"] = stop

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = LLMError(f"Ağ hatası: {e}")
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise last_err from e

            # Geçici hatalar (429 / 5xx) → yeniden dene; diğer hatalar → hemen düş.
            if resp.status_code != 200:
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_err = LLMError(f"API {resp.status_code}: {resp.text[:300]}")
                    if attempt < self.max_retries:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                raise LLMError(f"API {resp.status_code}: {resp.text[:500]}")

            data = resp.json()
            try:
                message = data["choices"][0]["message"]
                text = message["content"]
            except (KeyError, IndexError) as e:
                raise LLMError(f"Beklenmeyen yanıt biçimi: {data}") from e

            # `usage` her sağlayıcıda gelmeyebilir; yoksa 0 kabul ediyoruz.
            usage = data.get("usage") or {}
            return LLMResponse(
                text=text,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                # Qwen3 gibi "thinking" modelleri düşünmeyi ayrı alanda döner.
                reasoning=message.get("reasoning_content") or "",
            )

        raise last_err or LLMError("LLM çağrısı başarısız.")
