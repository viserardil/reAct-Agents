# Sıfırdan ReAct Agent

LangGraph/LangChain gibi hazır kütüphaneler **kullanmadan**, sadece HTTP çağrılarıyla
yazılmış bir ReAct (Reason + Act) agent'ı. LLM sağlayıcısı **bağımsızdır**:
OpenAI-uyumlu `/chat/completions` sunan herhangi bir sağlayıcıyla (HF Router, OpenAI,
Gemini, Groq, OpenRouter, yerel Ollama…) çalışır.

## ReAct nedir?

Model tek seferde cevap vermek yerine bir döngüde çalışır:

```
Question → Thought → Action → Observation → Thought → ... → Final Answer
```

- **Thought**: model ne yapacağını düşünür
- **Action + Action Input**: bir araç seçer
- **Observation**: aracı *biz* çalıştırıp sonucu geri veririz (model uyduramaz)
- Cevap bulununca **Final Answer** ile biter

## Dosya yapısı

| Dosya | Sorumluluk |
|-------|-----------|
| `react_agent/llm.py` | Sağlayıcı-bağımsız `/chat/completions` HTTP istemcisi (timeout + retry) |
| `react_agent/tools.py` | Araçlar (calculator, wikipedia) + kayıt sistemi |
| `react_agent/prompts.py` | Sistem prompt'u / ReAct formatı |
| `react_agent/agent.py` | Ana düşün-eylem-gözlem döngüsü |
| `main.py` | Giriş noktası |

## Kurulum & Çalıştırma (Windows PowerShell)

```powershell
py -m pip install -r requirements.txt

# En basit: HuggingFace token (varsayılan sağlayıcı HF Router)
$env:HF_TOKEN = "hf_xxxxxxxx"

# Model (varsayılan: Qwen/Qwen3.5-122B-A10B:deepinfra)
$env:HF_MODEL = "Qwen/Qwen3.5-122B-A10B:deepinfra"

py main.py "23 ile 19'un çarpımına 5 ekle"
```

### Farklı LLM sağlayıcısı (sağlayıcı-bağımsız)

`LLM_*` değişkenleri ayarlanırsa `HF_*`'ın yerine geçer (hiçbiri yoksa HF Router'a
düşer). Çoğu sağlayıcı OpenAI-uyumlu `/chat/completions` sunar; base'i `.../v1`
diye ver, istemci `/chat/completions`'ı ekler.

| Değişken | Açıklama |
|----------|----------|
| `LLM_API_KEY` | Sağlayıcının API anahtarı (fallback: `HF_TOKEN`, `OPENAI_API_KEY`, …). |
| `LLM_MODEL` | Model adı (ör. `gpt-4o-mini`, `gemini-2.5-flash`). |
| `LLM_BASE_URL` | Sağlayıcının OpenAI-uyumlu base URL'i. |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` | Tek çağrı zaman aşımı (sn, vars. 90) ve yeniden deneme (vars. 2). |

Örnek base_url'ler: OpenAI `https://api.openai.com/v1` · Gemini
`https://generativelanguage.googleapis.com/v1beta/openai/` · Groq
`https://api.groq.com/openai/v1` · yerel Ollama `http://localhost:11434/v1`.

> **Benchmark kıyası** için Plan-Execute tarafıyla **aynı** modeli kullan (ikisi de
> aynı sağlayıcı/model → tek değişken mimari kalır).

## Yeni araç eklemek

`tools.py` içine bir fonksiyon yazıp `@tool` ile kaydetmen yeter — agent kodu değişmez:

```python
@tool("saat", "Şu anki saati döner. Girdi gerekmez.")
def saat(_):
    from datetime import datetime
    return datetime.now().strftime("%H:%M")
```
