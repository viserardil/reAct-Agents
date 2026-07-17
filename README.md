# Sıfırdan ReAct Agent

LangGraph / LangChain gibi hazır agent kütüphaneleri **kullanmadan**, yalnızca ham
HTTP çağrılarıyla sıfırdan yazılmış bir **ReAct (Reason + Act)** agent'ı. Döngü,
prompt, çıktı ayrıştırma, araç yönetimi, bellek ve eval altyapısının tamamı elle
yazıldı — böylece "kaputun altında" ne olduğu birebir görülebiliyor.

Üstüne üç kullanım katmanı gelir:
- 🖥️ **CLI** — terminalden tek soru (`main.py`)
- 💬 **Web sohbet UI** — FastAPI + React, adım adım trace ve metriklerle (`chat_api.py`)
- 🧪 **Eval koşucusu** — HuggingFace dataset sorularını çalıştırıp standart JSON şemasına
  uygun sonuç üretir (`test/test.py`)

LLM sağlayıcısı **bağımsızdır**: OpenAI-uyumlu `/chat/completions` sunan herhangi bir
sağlayıcıyla (HF Router, OpenAI, Gemini, Groq, DeepInfra, OpenRouter, yerel Ollama…)
çalışır.

---

## ReAct nedir?

Model tek seferde cevap vermek yerine bir **düşün → eylem → gözlem** döngüsünde çalışır:

```
Question → Thought → Action → Action Input → Observation → Thought → … → Final Answer
```

- **Thought** — model ne yapacağını düşünür
- **Action + Action Input** — bir araç seçer ve girdisini verir
- **Observation** — aracı *biz* çalıştırıp sonucu geri veririz (model uyduramaz)
- Döngü gerektiği kadar tekrarlar; cevap bulununca **Final Answer** ile biter

Kilit tasarım kararı: modeli `Observation:` dizisinde **durdururuz** ki gözlemi kendisi
uydurmasın — aracı gerçekten biz çalıştırıp sonucu prompt'a ekleriz.

---

## Mimari

```
Staj_react_scratch/
├── react_agent/
│   ├── agent.py        # Ana düşün-eylem-gözlem döngüsü + RunResult/TraceStep + bellek
│   ├── llm.py          # Sağlayıcı-bağımsız /chat/completions HTTP istemcisi (retry'li)
│   ├── tools.py        # 18 araç + @tool kayıt sistemi (registry)
│   ├── prompts.py      # Sistem prompt'u (ReAct formatı) + geçmiş (bellek) enjeksiyonu
│   └── run_schema.py   # RunResult → standart eval şeması (v2.0.0) dönüştürücü
├── main.py             # CLI giriş noktası
├── chat_api.py         # FastAPI sohbet sunucusu (Web UI'ı ve /api/chat, /api/reset servis eder)
├── index.html          # Build gerektirmeyen React arayüzü (trace + metrikler)
├── test/
│   ├── test.py         # Dataset query'leriyle canlı eval koşucusu (CLI)
│   ├── a.json          # RunResult JSON Şeması (v2.0.0) — çıktı bununla doğrulanır
│   └── results/        # Koşu çıktıları (özet + şema + ilerleme akışı)
└── pyproject.toml      # Bağımlılıklar (uv ile yönetilir)
```

### Döngü nasıl çalışır (`agent.py`)

1. Talimatlar `system` rolünde, çalışma izi (scratchpad) `user` rolünde gönderilir.
2. Model `Observation:`'da durdurulur → tek bir Thought+Action üretir.
3. Çıktı ayrıştırılır: **Final Answer** mı, yoksa **Action** mı?
4. Action ise: araç çalıştırılır, gözlem scratchpad'e eklenir, 2'ye dönülür.
5. Her tur; **adım adım trace** (thought / reasoning / action / observation), **token**,
   **süre**, **araç latency'si** ve **scratchpad**'i toplar → `RunResult`.

---

## Araç kataloğu (18 araç)

| Araç | Ne yapar |
|------|----------|
| `calculator` | Güvenli matematik (AST tabanlı, `eval` yok) |
| `web_search` | Web araması (Tavily API) |
| `resolve_ticker` | Şirket adı → borsa sembolü ("Aselsan" → ASELS.IS) |
| `get_current_stock_price` | Güncel hisse fiyatı |
| `get_company_info` | Şirket profili (sektör, ülke, çalışan, özet) |
| `get_historical_stock_prices` | Geçmiş fiyatlar (dönem/aralık) |
| `get_stock_fundamentals` | Temel veriler (F/K, PD/DD, EPS, beta, 52h) |
| `get_income_statements` | Yıllık gelir tablosu |
| `get_quarterly_income_statements` | Çeyreklik gelir tablosu |
| `get_balance_sheet` | Yıllık bilanço |
| `get_cash_flow` | Yıllık nakit akışı |
| `get_key_financial_ratios` | Kâr marjları, ROE/ROA, borç/özkaynak |
| `get_analyst_recommendations` | Analist tavsiyeleri + hedef fiyatlar |
| `get_company_news` | Son haberler |
| `get_technical_indicators` | SMA/EMA, RSI, **MACD**, **Bollinger** |
| `compare_stocks` | İki hisseyi yan yana karşılaştırır |
| `plot_chart` | Hisse fiyat/hasılat grafiğini PNG kaydeder |
| `visualize_data` | Verilen ham veriyi (bar/line/pie) PNG grafiğe döker |

> Finans araçları **yfinance**, web araması **Tavily**, grafikler **matplotlib** kullanır.
> TR borsası da desteklenir (`.IS` uzantısı → TRY). Araçlar hatada çökmez; agent'a
> temiz bir hata mesajı (Observation) döner.

---

## Öne çıkan özellikler

- **Sağlayıcı-bağımsız LLM** — HF, OpenAI, Gemini, Groq, Ollama… (env ile)
- **Çok-turlu bellek** — `thread_id` başına önceki soru/cevaplar bağlama enjekte edilir
- **Reasoning yakalama** — Qwen3 gibi "thinking" modellerinin `reasoning_content`'i ayrı tutulur
- **Reasoning sızıntısı koruması** — `system/user` rol ayrımı + cevap temizleme kelepçesi
- **Markdown-güvenli** — rapor cevapları (başlık/tablo/liste) korunur
- **Standart eval çıktısı** — her koşum `test/a.json` şemasına (v2.0.0) uyar; harici scorer okuyabilir
- **Ağ dayanıklılığı** — 429/5xx/timeout'ta otomatik yeniden deneme

---

## Kurulum

Bağımlılıklar [uv](https://docs.astral.sh/uv/) ile yönetilir.

```powershell
uv sync                     # sanal ortam + tüm bağımlılıklar
uv sync --group dev         # + geliştirme/test paketleri (httpx)
```

Proje köküne bir `.env` dosyası oluştur:

```dotenv
# LLM (HuggingFace örneği) — LLM_* verirsen HF_*'ın yerine geçer
HF_TOKEN=hf_xxxxxxxx
# Sağlayıcı pini (":deepinfra") EKLEME — Router en hızlıyı seçsin (aksi halde
# o sağlayıcı yavaşsa çağrılar 10 kata kadar uzayabilir).
HF_MODEL=Qwen/Qwen3.5-122B-A10B

# Web araması için (opsiyonel)
TAVILY_API_KEY=tvly-xxxxxxxx
```

> `.env`, `.gitignore` ile hariç tutulur — anahtarlar repoya gitmez.

---

## Çalıştırma

### 1) CLI (tek soru)

```powershell
uv run python main.py "Apple hissesi bugün kaç dolar, RSI'ına göre aşırı alım bölgesinde mi?"
```

### 2) Web sohbet UI

```powershell
uv run python chat_api.py        # -> http://127.0.0.1:8000
```

Tarayıcıda soru yaz → cevap + metrikler (adım, araç, token, süre) + açılabilir
**trace** (her adımın thought/action/observation'ı). Aynı sohbet önceki turları
hatırlar; 🗑 Temizle belleği sıfırlar. Kod değişince sunucu otomatik yenilenir (reload).

### 3) Eval koşucusu (dataset)

`sccaglayanworkacc/equity-research-agentic-eval` dataset'inin `query` sorularını çalıştırır:

```powershell
uv run python test/test.py --list --limit 10        # query'leri listele (LLM çağırmaz)
uv run python test/test.py --limit 5 --validate     # ilk 5 soruyu çalıştır + şema doğrula
uv run python test/test.py --index 0 3 7            # sadece bu index'ler
```

Çıktılar `test/results/` altına yazılır:
- `results_<ad>.json` — okunası özet: cevap, metrikler, **trace** (reasoning), **scratchpad**
- `results_<ad>_schema.json` — `a.json` şemasına uygun RunResult dizisi (asıl eval çıktısı)
- `progress_<ad>.jsonl` — anlık ilerleme akışı (uzun koşuda kayıp olmaz)

Faydalı bayraklar: `--model`, `--max-steps`, `--split`, `--query-field`, `--temperature`,
`--framework`, `--no-progress`.

---

## Sağlayıcı-bağımsız LLM ayarı

`LLM_*` değişkenleri ayarlanırsa `HF_*`'ın yerine geçer; hiçbiri yoksa HF Router'a düşülür.
Çoğu sağlayıcı OpenAI-uyumlu `/chat/completions` sunar — base'i `.../v1` diye ver,
istemci `/chat/completions`'ı ekler.

| Değişken | Açıklama |
|----------|----------|
| `LLM_API_KEY` | API anahtarı (fallback: `HF_TOKEN`, `OPENAI_API_KEY`, `GROQ_API_KEY`…) |
| `LLM_MODEL` | Model adı (ör. `gpt-4o-mini`, `gemini-2.5-flash`) |
| `LLM_BASE_URL` | Sağlayıcının OpenAI-uyumlu base URL'i |
| `LLM_TIMEOUT` | Tek çağrı zaman aşımı (sn, varsayılan 90) |
| `LLM_MAX_RETRIES` | Yeniden deneme sayısı (varsayılan 2) |

Örnek base URL'ler: OpenAI `https://api.openai.com/v1` · Gemini
`https://generativelanguage.googleapis.com/v1beta/openai/` · Groq
`https://api.groq.com/openai/v1` · yerel Ollama `http://localhost:11434/v1`.

**Azure OpenAI** (deployment + api-version + `api-key` header ile OpenAI'den ayrı):
```env
LLM_PROVIDER=azure_openai
AZURE_OPENAI_ENDPOINT=https://<kaynak-adin>.openai.azure.com
AZURE_OPENAI_API_KEY=<key>
OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT=<deployment-adin>   # Azure'da model değil, DEPLOYMENT adı
```

---

## Yeni araç eklemek

`tools.py` içine bir fonksiyon yazıp `@tool` ile kaydetmen yeter — **agent ve prompt
kodu değişmez** (araç otomatik olarak sistem prompt'una eklenir):

```python
@tool("saat", "Şu anki saati döner. Girdi gerekmez.")
def saat(_):
    from datetime import datetime
    return datetime.now().strftime("%H:%M")
```

yfinance tabanlı araçlar için `@finance_tool` dekoratörü `yf`'i enjekte eder ve
hata yönetimini üstlenir.

---

## Eval şeması (`test/a.json`)

Her koşum, **RunResult v2.0.0** JSON Şemasına birebir uyan bir nesneye çevrilir
(`run_schema.py`). Şema "olgu tutar, skor tutmaz": koşumun ne yaptığını
(adımlar, araç çağrıları, token, latency, timestamp, tam trace) kaydeder; doğru/yanlış
kararını harici bir scorer ya da sen verirsin. `--validate` ile çıktı `a.json`'a karşı
`jsonschema` ile doğrulanır.
