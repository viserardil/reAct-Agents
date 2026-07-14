
import os
import time
from dataclasses import dataclass

import requests

# SAĞLAYICI-BAĞIMSIZ istemci: OpenAI-uyumlu /chat/completions konuşan HERHANGİ bir
# sağlayıcıyla çalışır (HF Router, OpenAI, Gemini-compat, Groq, Together, OpenRouter,
# DeepInfra, yerel Ollama/vLLM...). Base URL, anahtar ve model env'den okunur;
# LLM_* ayarlıysa HF_*'ın yerine geçer, hiçbiri yoksa HF Router'a düşer.
DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
# Sağlayıcı PİN'İ YOK (":deepinfra" gibi): HF Router en hızlı/müsait sağlayıcıyı
# seçsin. Belirli bir sağlayıcıya pinlemek o an yavaşsa 10 kat gecikme yaratabilir.
DEFAULT_MODEL = "Qwen/Qwen3.5-122B-A10B"


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
        # Azure OpenAI standart OpenAI'den ayrı: deployment-tabanlı URL, zorunlu
        # api-version query param ve 'api-key' header (Bearer değil).
        self._azure = (os.environ.get("LLM_PROVIDER", "").strip().lower() in ("azure", "azure_openai")
                       or bool(os.environ.get("AZURE_OPENAI_ENDPOINT")))

        # Anahtar: parametre > (Azure ise AZURE_OPENAI_API_KEY) > LLM_API_KEY > HF_TOKEN > ...
        if self._azure:
            self.api_key = api_key or _first("AZURE_OPENAI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")
        else:
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

        if self._azure:
            # model = Azure DEPLOYMENT adı; URL deployment + api-version içerir.
            self.model = model or _first("AZURE_OPENAI_DEPLOYMENT", "LLM_MODEL", "HF_MODEL") or DEFAULT_MODEL
            endpoint = (os.environ.get("AZURE_OPENAI_ENDPOINT") or base_url or _first("LLM_BASE_URL") or "").rstrip("/")
            api_version = _first("OPENAI_API_VERSION", "AZURE_OPENAI_API_VERSION") or "2024-10-21"
            self.url = f"{endpoint}/openai/deployments/{self.model}/chat/completions?api-version={api_version}"
        else:
            self.model = model or _first("LLM_MODEL", "HF_MODEL") or DEFAULT_MODEL
            # Base URL (LLM_BASE_URL > HF_BASE_URL > varsayılan). Uçtaki /chat/completions
            # yoksa ekleriz; sağlayıcılar base'i genelde .../v1 diye verir.
            base = (base_url or _first("LLM_BASE_URL", "HF_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
            self.url = base if base.endswith("/chat/completions") else base + "/chat/completions"

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout if timeout is not None else float(os.environ.get("LLM_TIMEOUT", "90"))
        self.max_retries = max_retries if max_retries is not None else int(os.environ.get("LLM_MAX_RETRIES", "2"))

        # "Thinking" (reasoning) kontrolü. Qwen3 gibi modeller cevaptan önce uzun bir
        # düşünme üretir; ReAct'te bu gereksiz (kendi akıl yürütmemizi yapıyoruz) ve
        # bazı sağlayıcılar onu `content`'e döküp cevaba SIZDIRIYOR. Bu yüzden Qwen3'te
        # varsayılan olarak KAPATIRIZ. Diğer modellere dokunmayız (param göndermeyiz).
        #   None  → param gönderme (sağlayıcı varsayılanı)
        #   False → chat_template_kwargs.enable_thinking=false gönder
        #   True  → enable_thinking=true gönder
        env_think = os.environ.get("LLM_ENABLE_THINKING")
        if env_think is not None:
            self.enable_thinking = env_think.strip().lower() in ("1", "true", "yes", "on")
        elif "qwen3" in (self.model or "").lower():
            self.enable_thinking = False
        else:
            self.enable_thinking = None

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
        # Azure → 'api-key' header; diğerleri → 'Authorization: Bearer'.
        if self._azure:
            headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        else:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # Qwen3 vb. için düşünmeyi aç/kapat (reasoning sızıntısını önler).
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
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
                        # 429'da sunucunun Retry-After'ına saygı (yoksa artan bekleme;
                        # TPM penceresi ~60s olduğu için 429'da daha uzun bekle).
                        ra = resp.headers.get("retry-after", "")
                        if ra.replace(".", "", 1).isdigit():
                            wait = float(ra)
                        elif resp.status_code == 429:
                            wait = 8 * (attempt + 1)
                        else:
                            wait = 1.5 * (attempt + 1)
                        time.sleep(min(wait, 65))
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
