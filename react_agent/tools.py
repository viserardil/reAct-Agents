"""
Araçlar (tools) — agent'ın dış dünyaya "eylem" (Act) yapabildiği fonksiyonlar.

Her araç basit bir Python fonksiyonudur: string girdi alır, string çıktı verir.
Bir kayıt (registry) ile isimlerini agent'a tanıtırız. Yeni bir yetenek eklemek
= yeni bir fonksiyon yazıp @tool ile kaydetmek. Agent kodu değişmez.
"""

import ast
import logging
import operator
import os
import urllib.parse
import requests

# yfinance geçersiz sembolde konsola 404/uyarı basar; bunu susturuyoruz
# (araçlar zaten temiz bir hata mesajı döndürüyor).
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# İsim -> araç bilgisi haritası. Agent bu sözlüğe bakarak araçları bulur.
TOOLS = {}


def tool(name, description):
    """Bir fonksiyonu araç olarak kaydeden dekoratör."""
    def wrapper(func):
        TOOLS[name] = {"func": func, "description": description}
        return func
    return wrapper


# --- Güvenli hesap makinesi -------------------------------------------------
# eval() TEHLİKELİDİR (rasgele kod çalıştırır). Bunun yerine ifadeyi AST'ye
# çevirip yalnızca izin verdiğimiz matematik operatörlerini yürütüyoruz.
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):  # sayı sabiti
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Sadece sayılar desteklenir")
    if isinstance(node, ast.BinOp):     # a + b gibi ikili işlem
        return _OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):   # -a gibi tekli işlem
        return _OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("İzin verilmeyen ifade")


@tool("calculator", "Matematiksel ifadeyi hesaplar. Girdi: '2 * (3 + 4)' gibi.")
def calculator(expr):
    try:
        tree = ast.parse(expr, mode="eval")
        return str(_safe_eval(tree))
    except Exception as e:
        return f"Hesaplama hatası: {e}"




@tool("web_search", "Güncel bilgi için web'de arama yapar (Tavily). Girdi: arama sorgusu, ör. 'en son Fed faiz kararı'.")
def web_search(query):
    # Tavily REST API'sine ham HTTP çağrısı (SDK yok). Anahtar env'den okunur.
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "TAVILY_API_KEY ayarlı değil (ortam değişkeni olarak ver)."
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query.strip(),
                "max_results": 5,
                "search_depth": "basic",
                "include_answer": True,   # Tavily'nin ürettiği kısa özet
            },
            timeout=25,
        )
    except requests.RequestException as e:
        return f"Ağ hatası: {e}"

    if resp.status_code != 200:
        return f"Tavily hatası {resp.status_code}: {resp.text[:300]}"

    data = resp.json()
    lines = []
    if data.get("answer"):
        lines.append(f"Özet: {data['answer']}")
    for i, r in enumerate(data.get("results", []), 1):
        snippet = (r.get("content") or "")[:200].replace("\n", " ")
        lines.append(f"{i}. {r.get('title', '(başlık yok)')} — {snippet}\n   {r.get('url', '')}")
    return "\n".join(lines) if lines else "Sonuç bulunamadı."


# --- Borsa araçları (yfinance) ----------------------------------------------
# yfinance'i fonksiyon içinde import ediyoruz: kütüphane kurulu değilse çekirdek
# agent yine çalışır, sadece bu araçlar hata mesajı döner.

def _yf():
    try:
        import yfinance as yf
        return yf
    except ImportError:
        return None


def finance_tool(name, description):
    """yfinance aracı dekoratörü: yf'yi enjekte eder, kütüphane yoksa /
    beklenmeyen hatada temiz bir mesaj döndürür. Böylece her araçta aynı
    try/except iskeletini tekrar yazmayız."""
    def deco(func):
        def wrapped(arg):
            yf = _yf()
            if yf is None:
                return "yfinance kurulu değil (uv add yfinance)."
            try:
                return func(yf, arg)
            except Exception as e:
                return f"{name} hatası: {e}"
        TOOLS[name] = {"func": wrapped, "description": description}
        return wrapped
    return deco


def _num(x, nd=2):
    """Sayıyı güvenle biçimlendir; None/eksikse '?' döndür."""
    if isinstance(x, (int, float)):
        return f"{x:,.{nd}f}" if abs(x) < 1e12 else f"{x:,.0f}"
    return "?"


def _sym(arg):
    """Girdinin ilk kelimesini sembol olarak al (büyük harf)."""
    parts = arg.strip().split()
    return parts[0].upper() if parts else ""


@finance_tool("get_current_stock_price", "Bir şirketin güncel hisse fiyatını döner. Girdi: sembol, ör. 'AAPL' veya 'THYAO.IS'.")
def get_current_stock_price(yf, symbol):
    sym = _sym(symbol)
    fi = yf.Ticker(sym).fast_info
    price = fi.get("lastPrice") if hasattr(fi, "get") else getattr(fi, "last_price", None)
    if price is None:
        return f"'{sym}' için fiyat bulunamadı (sembol yanlış olabilir)."
    cur = (fi.get("currency") if hasattr(fi, "get") else getattr(fi, "currency", "")) or ""
    return f"{sym}: {price:.2f} {cur}".strip()


@finance_tool("get_company_info", "Bir şirket hakkında ayrıntılı bilgi (isim, sektör, ülke, çalışan sayısı, özet). Girdi: sembol, ör. 'MSFT'.")
def get_company_info(yf, symbol):
    sym = _sym(symbol)
    info = yf.Ticker(sym).info
    name = info.get("longName") or info.get("shortName")
    if not name:
        return f"'{sym}' için bilgi bulunamadı."
    summary = (info.get("longBusinessSummary") or "")[:400]
    lines = [
        f"İsim: {name} ({sym})",
        f"Sektör/Endüstri: {info.get('sector', '?')} / {info.get('industry', '?')}",
        f"Ülke: {info.get('country', '?')}, Çalışan: {info.get('fullTimeEmployees', '?')}",
        f"Piyasa değeri: {_num(info.get('marketCap'))} {info.get('currency', '')}",
        f"Web: {info.get('website', '?')}",
    ]
    if summary:
        lines.append(f"Özet: {summary}...")
    return "\n".join(lines)


@finance_tool("get_historical_stock_prices", "Geçmiş hisse fiyatlarını döner. Girdi: 'SEMBOL [DÖNEM] [ARALIK]', ör. 'AAPL 1mo 1d' (dönem: 5d,1mo,6mo,1y; aralık: 1d,1wk,1mo).")
def get_historical_stock_prices(yf, query):
    parts = query.strip().split()
    sym = parts[0].upper() if parts else ""
    period = parts[1] if len(parts) > 1 else "1mo"
    interval = parts[2] if len(parts) > 2 else "1d"
    hist = yf.Ticker(sym).history(period=period, interval=interval)
    if hist.empty:
        return f"'{sym}' için '{period}/{interval}' veri yok."
    tail = hist.tail(10)[["Open", "High", "Low", "Close", "Volume"]].round(2)
    tail.index = tail.index.strftime("%Y-%m-%d")
    return f"{sym} ({period}, {interval}) — son {len(tail)} kayıt:\n{tail.to_string()}"


@finance_tool("get_stock_fundamentals", "Bir hissenin temel verileri: piyasa değeri, F/K, PD/DD, EPS, temettü, beta, 52h yüksek/düşük. Girdi: sembol.")
def get_stock_fundamentals(yf, symbol):
    sym = _sym(symbol)
    info = yf.Ticker(sym).info
    if not (info.get("longName") or info.get("shortName")):
        return f"'{sym}' için veri bulunamadı."
    dy = info.get("dividendYield")
    # Not: bu yfinance sürümünde dividendYield zaten yüzde olarak gelir (0.34 = %0.34).
    return "\n".join([
        f"{sym} temel veriler:",
        f"Piyasa değeri: {_num(info.get('marketCap'))} {info.get('currency', '')}",
        f"F/K (trailing/forward): {_num(info.get('trailingPE'))} / {_num(info.get('forwardPE'))}",
        f"PD/DD: {_num(info.get('priceToBook'))}, EPS: {_num(info.get('trailingEps'))}",
        f"Temettü verimi: {f'{dy:.2f}%' if isinstance(dy, (int, float)) else '?'}",
        f"Beta: {_num(info.get('beta'))}",
        f"52h yüksek/düşük: {_num(info.get('fiftyTwoWeekHigh'))} / {_num(info.get('fiftyTwoWeekLow'))}",
    ])


@finance_tool("get_income_statements", "Bir şirketin yıllık gelir tablosu (hasılat, brüt kâr, faaliyet kârı, net kâr). Girdi: sembol.")
def get_income_statements(yf, symbol):
    sym = _sym(symbol)
    stmt = yf.Ticker(sym).income_stmt
    if stmt is None or stmt.empty:
        return f"'{sym}' için gelir tablosu yok."
    rows = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income"]
    out = [f"{sym} gelir tablosu (yıllık, para birimi ham):"]
    for col in stmt.columns[:4]:  # en yeni 4 yıl
        year = col.strftime("%Y") if hasattr(col, "strftime") else str(col)
        parts = []
        for r in rows:
            if r in stmt.index:
                parts.append(f"{r}={_num(stmt.loc[r, col], 0)}")
        out.append(f"{year}: " + ", ".join(parts))
    return "\n".join(out)


@finance_tool("get_key_financial_ratios", "Bir şirketin temel finansal oranları: kâr marjları, ROE/ROA, borç/özkaynak, cari oran. Girdi: sembol.")
def get_key_financial_ratios(yf, symbol):
    sym = _sym(symbol)
    info = yf.Ticker(sym).info
    if not (info.get("longName") or info.get("shortName")):
        return f"'{sym}' için oran verisi yok."
    def pct(k):
        v = info.get(k)
        return f"{v * 100:.2f}%" if isinstance(v, (int, float)) else "?"
    return "\n".join([
        f"{sym} finansal oranlar:",
        f"Brüt marj: {pct('grossMargins')}, Faaliyet marjı: {pct('operatingMargins')}, Net marj: {pct('profitMargins')}",
        f"ROE: {pct('returnOnEquity')}, ROA: {pct('returnOnAssets')}",
        f"Borç/Özkaynak: {_num(info.get('debtToEquity'))}, Cari oran: {_num(info.get('currentRatio'))}, Nakit oranı: {_num(info.get('quickRatio'))}",
        f"PD/DD: {_num(info.get('priceToBook'))}, PEG: {_num(info.get('pegRatio'))}",
    ])


@finance_tool("get_analyst_recommendations", "Bir hisse için analist tavsiyeleri ve hedef fiyatlar. Girdi: sembol.")
def get_analyst_recommendations(yf, symbol):
    sym = _sym(symbol)
    info = yf.Ticker(sym).info
    key = info.get("recommendationKey", "?")
    n = info.get("numberOfAnalystOpinions", "?")
    lines = [
        f"{sym} analist görüşü: {key} ({n} analist)",
        f"Hedef fiyat — ort: {_num(info.get('targetMeanPrice'))}, "
        f"yüksek: {_num(info.get('targetHighPrice'))}, düşük: {_num(info.get('targetLowPrice'))}",
    ]
    try:
        rec = yf.Ticker(sym).recommendations
        if rec is not None and not rec.empty:
            row = rec.iloc[0]
            lines.append(
                f"Dağılım: strongBuy={row.get('strongBuy', '?')}, buy={row.get('buy', '?')}, "
                f"hold={row.get('hold', '?')}, sell={row.get('sell', '?')}, strongSell={row.get('strongSell', '?')}"
            )
    except Exception:
        pass
    return "\n".join(lines)


@finance_tool("get_company_news", "Bir şirketle ilgili son haberleri döner. Girdi: sembol.")
def get_company_news(yf, symbol):
    sym = _sym(symbol)
    news = yf.Ticker(sym).news or []
    if not news:
        return f"'{sym}' için haber bulunamadı."
    out = [f"{sym} son haberler:"]
    for item in news[:5]:
        # yfinance iki farklı biçim kullanabiliyor: düz ya da 'content' altında.
        content = item.get("content", item)
        title = content.get("title") or item.get("title") or "(başlık yok)"
        pub = (content.get("provider", {}) or {}).get("displayName") or item.get("publisher") or ""
        out.append(f"- {title}" + (f" [{pub}]" if pub else ""))
    return "\n".join(out)


@finance_tool("get_technical_indicators", "Hisse için teknik göstergeler (SMA20/50, EMA20, RSI14, MACD, Bollinger) hesaplar. Girdi: 'SEMBOL [DÖNEM]', ör. 'AAPL 6mo'.")
def get_technical_indicators(yf, query):
    parts = query.strip().split()
    sym = parts[0].upper() if parts else ""
    period = parts[1] if len(parts) > 1 else "6mo"
    hist = yf.Ticker(sym).history(period=period)
    if hist.empty or len(hist) < 15:
        return f"'{sym}' için yeterli veri yok ({period})."
    close = hist["Close"]
    last = close.iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = (100 - 100 / (1 + rs)).iloc[-1]
    # MACD(12,26,9): hızlı-yavaş EMA farkı + sinyal çizgisi
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd, signal = macd_line.iloc[-1], signal_line.iloc[-1]
    macd_hist = macd - signal
    macd_yon = "al sinyali (MACD>Sinyal)" if macd > signal else "sat sinyali (MACD<Sinyal)"
    # Bollinger Bantları (20, 2σ)
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = (mid + 2 * std).iloc[-1]
    lower = (mid - 2 * std).iloc[-1]
    trend = "yukarı" if last > sma50 else "aşağı"
    return "\n".join([
        f"{sym} teknik göstergeler ({period}):",
        f"Son kapanış: {last:.2f}",
        f"SMA20: {_num(sma20)}, SMA50: {_num(sma50)}, EMA20: {_num(ema20)}",
        f"RSI(14): {_num(rsi)} ({'aşırı alım' if rsi > 70 else 'aşırı satım' if rsi < 30 else 'nötr'})",
        f"MACD: {_num(macd)}, Sinyal: {_num(signal)}, Histogram: {_num(macd_hist)} → {macd_yon}",
        f"Bollinger: alt {_num(lower)} / orta {_num(mid.iloc[-1])} / üst {_num(upper)}",
        f"Fiyat SMA50'nin {trend}sında (trend: {trend})",
    ])


@finance_tool("get_quarterly_income_statements", "Çeyreklik gelir tablosu (hasılat, brüt/faaliyet/net kâr). Girdi: sembol.")
def get_quarterly_income_statements(yf, symbol):
    sym = _sym(symbol)
    stmt = yf.Ticker(sym).quarterly_income_stmt
    if stmt is None or stmt.empty:
        return f"'{sym}' için çeyreklik gelir tablosu yok."
    rows = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income"]
    out = [f"{sym} çeyreklik gelir tablosu:"]
    for col in stmt.columns[:4]:   # son 4 çeyrek
        q = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
        parts = [f"{r}={_num(stmt.loc[r, col], 0)}" for r in rows if r in stmt.index]
        out.append(f"{q}: " + ", ".join(parts))
    return "\n".join(out)


@finance_tool("get_balance_sheet", "Yıllık bilanço özeti (varlık, borç, özkaynak, nakit). Girdi: sembol.")
def get_balance_sheet(yf, symbol):
    sym = _sym(symbol)
    bs = yf.Ticker(sym).balance_sheet
    if bs is None or bs.empty:
        return f"'{sym}' için bilanço verisi yok."
    rows = ["Total Assets", "Total Liabilities Net Minority Interest",
            "Stockholders Equity", "Total Debt", "Cash And Cash Equivalents"]
    out = [f"{sym} bilanço (yıllık):"]
    for col in bs.columns[:3]:
        y = col.strftime("%Y") if hasattr(col, "strftime") else str(col)
        parts = [f"{r}={_num(bs.loc[r, col], 0)}" for r in rows if r in bs.index]
        out.append(f"{y}: " + ", ".join(parts))
    return "\n".join(out)


@finance_tool("get_cash_flow", "Yıllık nakit akışı (faaliyet, yatırım, finansman, serbest nakit). Girdi: sembol.")
def get_cash_flow(yf, symbol):
    sym = _sym(symbol)
    cf = yf.Ticker(sym).cashflow
    if cf is None or cf.empty:
        return f"'{sym}' için nakit akışı verisi yok."
    rows = ["Operating Cash Flow", "Investing Cash Flow",
            "Financing Cash Flow", "Free Cash Flow"]
    out = [f"{sym} nakit akışı (yıllık):"]
    for col in cf.columns[:3]:
        y = col.strftime("%Y") if hasattr(col, "strftime") else str(col)
        parts = [f"{r}={_num(cf.loc[r, col], 0)}" for r in rows if r in cf.index]
        out.append(f"{y}: " + ", ".join(parts))
    return "\n".join(out)


@finance_tool("compare_stocks", "İki hisseyi yan yana karşılaştırır (fiyat, F/K, PD/DD, ROE, piyasa değeri). Girdi: 'SEMBOL1 SEMBOL2', ör. 'AAPL MSFT'.")
def compare_stocks(yf, query):
    parts = query.replace(",", " ").split()
    if len(parts) < 2:
        return "İki sembol gir: örn. 'AAPL MSFT'."
    a, b = parts[0].upper(), parts[1].upper()

    def snap(sym):
        info = yf.Ticker(sym).info
        roe = info.get("returnOnEquity")
        return {
            "Fiyat": _num(info.get("currentPrice") or info.get("regularMarketPrice")),
            "F/K": _num(info.get("trailingPE")),
            "PD/DD": _num(info.get("priceToBook")),
            "ROE": f"{roe * 100:.2f}%" if isinstance(roe, (int, float)) else "?",
            "Piyasa Değeri": _num(info.get("marketCap")),
        }

    ra, rb = snap(a), snap(b)
    out = [f"Karşılaştırma: {a} vs {b}"]
    for k in ra:
        out.append(f"{k}: {a}={ra[k]}  |  {b}={rb[k]}")
    return "\n".join(out)


def _ascii_fold(text):
    """Türkçe karakterleri ASCII'ye indir (Yahoo araması 'Şişecam'ı bulamıyor,
    'Sisecam'ı bulabiliyor)."""
    table = str.maketrans("şŞıİğĞüÜöÖçÇ", "sSiIgGuUoOcC")
    return text.translate(table)


def _yf_search(yf, q):
    """yfinance sürümüne göre Search/Lookup ile sembol arar; quote listesi döner."""
    if hasattr(yf, "Search"):
        res = yf.Search(q, max_results=5)
        return getattr(res, "quotes", None) or (res.get("quotes") if isinstance(res, dict) else [])
    if hasattr(yf, "Lookup"):
        return yf.Lookup(q).all.head(5).reset_index().to_dict("records")
    return []


@finance_tool("resolve_ticker", "Şirket adından borsa sembolünü bulur. Girdi: şirket adı, ör. 'Şişecam' veya 'Coca Cola'.")
def resolve_ticker(yf, name):
    q = name.strip()
    try:
        quotes = _yf_search(yf, q)
        # Türkçe adlar Yahoo'da tutmayabilir; ASCII'ye indirip bir daha dene.
        if not quotes and _ascii_fold(q) != q:
            quotes = _yf_search(yf, _ascii_fold(q))
    except Exception as e:
        return f"Sembol araması başarısız ({e}). web_search ile deneyebilirsin."
    lines = []
    for it in list(quotes)[:5]:
        sym = it.get("symbol") or it.get("Symbol") or it.get("index")
        nm = it.get("shortname") or it.get("longname") or it.get("shortName") or it.get("name") or ""
        exch = it.get("exchange") or it.get("exchDisp") or ""
        if sym:
            lines.append(f"{sym} — {nm} ({exch})".strip())
    if lines:
        return f"'{q}' için bulunan semboller:\n" + "\n".join(lines)
    return f"'{q}' için sembol bulunamadı. web_search ile aramayı deneyebilirsin."


# Grafikler buraya kaydedilir (proje kökü / scratch / charts).
CHARTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", "charts"
)


@finance_tool("plot_chart", "Bir hisse için grafik çizip PNG kaydeder ve yolunu döner. Girdi: 'SEMBOL [DÖNEM] [TÜR]', TÜR: price|revenue (ör. 'AAPL 6mo price').")
def plot_chart(yf, query):
    import uuid
    import matplotlib
    matplotlib.use("Agg")           # başsız (GUI'siz) çizim
    import matplotlib.pyplot as plt

    # TÜR'ü (price/revenue) konumdan bağımsız yakala: model 'KO revenue' de
    # yazabilir 'KO 6mo price' de. price/revenue dışındaki token = dönem.
    tokens = query.strip().split()
    sym = tokens[0].upper() if tokens else ""
    kind, period = "price", "6mo"
    for tok in tokens[1:]:
        if tok.lower() in ("price", "revenue"):
            kind = tok.lower()
        else:
            period = tok

    t = yf.Ticker(sym)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    path = os.path.join(CHARTS_DIR, f"{sym}_{kind}_{period}_{uuid.uuid4().hex[:6]}.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    try:
        if kind == "revenue":
            stmt = t.income_stmt
            if stmt is None or stmt.empty or "Total Revenue" not in stmt.index:
                return f"'{sym}' için gelir verisi yok."
            rev = stmt.loc["Total Revenue"].dropna()[::-1]
            years = [c.strftime("%Y") if hasattr(c, "strftime") else str(c) for c in rev.index]
            ax.bar(years, rev.values / 1e9)
            ax.set_ylabel("Hasılat (milyar)")
            ax.set_title(f"{sym} Yıllık Hasılat")
        else:
            hist = t.history(period=period)
            if hist.empty:
                return f"'{sym}' için fiyat verisi yok ({period})."
            ax.plot(hist.index, hist["Close"])
            ax.set_ylabel("Kapanış")
            ax.set_title(f"{sym} Fiyat ({period})")
            ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, dpi=100)
    finally:
        plt.close(fig)
    return f"Grafik kaydedildi: {path}"


@tool(
    "visualize_data",
    "VERİLEN ham veriyle grafik (bar/line/pie) çizip PNG kaydeder ve yolunu döner. "
    "Hisse fiyatı için değil, elindeki sayısal veriyi görselleştirmek için kullan. "
    "Girdi: JSON nesnesi. Alanlar: 'x' (etiketler listesi), 'y' (sayı listesi), "
    "'kind' (bar|line|pie, opsiyonel), 'title' (opsiyonel), 'instruction' (opsiyonel; "
    "tür buradan da çıkarılır). x/y 'data' altında iç içe de olabilir. "
    "Örn: {\"kind\":\"bar\",\"x\":[\"2019\",\"2020\"],\"y\":[10,12],\"title\":\"Gelir\"}",
)
def visualize_data(spec):
    import json
    import uuid
    import matplotlib
    matplotlib.use("Agg")            # başsız (GUI'siz) çizim
    import matplotlib.pyplot as plt

    try:
        data = json.loads(spec)
    except Exception:
        return ('HATA: girdi JSON olmalı, ör. '
                '{"kind":"bar","x":["2019","2020"],"y":[10,12],"title":"Gelir"}')
    if not isinstance(data, dict):
        return "HATA: JSON bir nesne (dict) olmalı."

    # x/y doğrudan ya da 'data' altında iç içe olabilir.
    inner = data.get("data") if isinstance(data.get("data"), dict) else data
    x = inner.get("x") or inner.get("labels") or inner.get("categories")
    y = inner.get("y") or inner.get("values")
    if not isinstance(y, list) or not y:
        return "HATA: 'y' (sayısal değerler listesi) gerekli."
    if not isinstance(x, list) or not x:
        x = [str(i + 1) for i in range(len(y))]
    try:
        yv = [float(v) for v in y]
    except Exception:
        return "HATA: 'y' değerleri sayısal olmalı."
    xl = [str(v) for v in x]

    # Grafik türü: açık 'kind' ya da instruction/title'dan çıkar.
    kind = (data.get("kind") or data.get("type") or "").lower()
    hint = f"{data.get('instruction', '')} {data.get('title', '')}".lower()
    if kind not in ("bar", "line", "pie"):
        # Açık tür kelimeleri önce (ör. "bar chart ... trend" → bar, line değil).
        if "pie" in hint or "pasta" in hint:
            kind = "pie"
        elif "bar" in hint or "çubuk" in hint or "sütun" in hint:
            kind = "bar"
        elif "line" in hint or "çizgi" in hint:
            kind = "line"
        elif "trend" in hint:      # zayıf ipucu: yalnızca açık tür yoksa
            kind = "line"
        else:
            kind = "bar"
    title = str(data.get("title") or data.get("instruction") or "Grafik")[:100]

    os.makedirs(CHARTS_DIR, exist_ok=True)
    path = os.path.join(CHARTS_DIR, f"viz_{kind}_{uuid.uuid4().hex[:6]}.png")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    try:
        if kind == "pie":
            ax.pie(yv, labels=xl, autopct="%1.1f%%")
        elif kind == "line":
            ax.plot(xl, yv, marker="o")
            ax.grid(True, alpha=0.3)
        else:
            ax.bar(xl, yv)
        ax.set_title(title)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, dpi=100)
    finally:
        plt.close(fig)
    return f"Grafik kaydedildi ({kind}, {len(yv)} nokta): {path}"


def render_tool_descriptions():
    """Sistem prompt'una gömmek için araç listesini metne çevirir."""
    return "\n".join(f"- {name}: {info['description']}" for name, info in TOOLS.items())


def run_tool(name, tool_input):
    """Adı verilen aracı çalıştırır; yoksa hata mesajı döner."""
    name = name.strip()
    if name not in TOOLS:
        return f"HATA: '{name}' diye bir araç yok. Mevcut araçlar: {list(TOOLS)}"
    return TOOLS[name]["func"](tool_input)
