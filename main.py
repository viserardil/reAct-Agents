"""
Terminalden hızlı test için giriş noktası (Web UI olmadan).

Kullanım (PowerShell):
    # .env içinde LLM_API_KEY / HF_TOKEN olmalı
    uv run python main.py "Apple hissesi bugün kaç dolar?"
    uv run python main.py "23 ile 19'un çarpımına 5 ekle"
    uv run python main.py            # argümansız: örnek soru
"""

import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from react_agent import ReActAgent


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "Apple hissesinin güncel fiyatı nedir ve RSI'ına göre aşırı alım bölgesinde mi?"

    agent = ReActAgent(verbose=True)   # verbose: adımları terminale bas
    print(f"Soru: {question}\n")
    result = agent.run(question)
    print(f"\nCevap: {result.answer}")
    print(f"(durum={result.status}, adım={result.steps}, araç={result.tool_calls}, "
          f"token={result.total_tokens}, süre={result.elapsed_seconds:.1f}sn)")


if __name__ == "__main__":
    main()
