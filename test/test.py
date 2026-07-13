"""ReAct ajanı için CANLI eval koşucusu — HuggingFace dataset'inden test case'leri.

Test case'leri "sccaglayanworkacc/equity-research-agentic-eval" dataset'inin
`query` sorularından okur, her birini gerçek bir HF Inference modeline karşı
çalıştırır ve sonucu test/a.json'daki v2.0.0 RunResult şemasına uygun JSON olarak
kaydeder. Beklenti/doğrulama YOK: olgu (trace + metrik) toplar; doğru/yanlış
değerlendirmesini sen çıktı JSON'undan yaparsın.

Kurulum:
    uv sync                                  # bağımlılıklar (datasets, jsonschema ...)
    .env içinde HF_TOKEN (ve isteğe bağlı HF_MODEL)

Çalıştırma (terminalden):
    uv run python test/test.py --limit 5            # ilk 5 query
    uv run python test/test.py --index 0 3 7        # sadece 0,3,7. query'ler
    uv run python test/test.py --list               # query'leri listele, çalıştırma
    uv run python test/test.py --limit 5 --validate # çıktıyı a.json şemasıyla doğrula
    uv run python test/test.py --version denemem     # çıktı dosya etiketi

HF_TOKEN yoksa hata verir. Çıktılar test/results/ altına yazılır.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# .env'i erken yükle ki HF_TOKEN/HF_MODEL hem burada hem ajanda okunsun.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Proje kökünü import edilebilir yap (test/ dizininden doğrudan çalıştırma için).
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATASET_NAME = "sccaglayanworkacc/equity-research-agentic-eval"
SCHEMA_PATH = ROOT_DIR / "test" / "a.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "test" / "results"
# Query metnini bu alan adlarından birinde arar (dataset şeması bilinmiyorsa).
QUERY_FIELD_CANDIDATES = ("query", "question", "prompt", "input", "task")


# --- Test case modeli -------------------------------------------------------


@dataclass
class Case:
    index: int
    case_id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


# --- Dataset yükleme --------------------------------------------------------


def _pick_query_field(columns: list[str], override: str | None) -> str:
    if override:
        if override not in columns:
            raise SystemExit(f"'{override}' alanı dataset'te yok. Mevcut alanlar: {columns}")
        return override
    for cand in QUERY_FIELD_CANDIDATES:
        if cand in columns:
            return cand
    raise SystemExit(
        f"Query alanı bulunamadı. Mevcut alanlar: {columns}. "
        f"--query-field ile elle belirt."
    )


def load_cases(dataset_name: str, split: str | None, query_field: str | None,
               limit: int | None) -> list[Case]:
    """Dataset'i indirir ve query sorularını Case listesine çevirir."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name)  # HF_TOKEN env'den otomatik okunur
    # split seçimi: verilmişse onu, yoksa ilk mevcut split'i kullan.
    if hasattr(ds, "keys"):  # DatasetDict
        available = list(ds.keys())
        chosen = split or available[0]
        if chosen not in ds:
            raise SystemExit(f"'{chosen}' split'i yok. Mevcut split'ler: {available}")
        data = ds[chosen]
    else:  # zaten tek Dataset
        data = ds

    columns = list(data.column_names)
    qfield = _pick_query_field(columns, query_field)
    # case_id için makul bir alan ara (yoksa index kullanılır).
    id_field = next((c for c in ("id", "case_id", "task_id", "ticker") if c in columns), None)

    cases: list[Case] = []
    n = len(data) if limit is None else min(limit, len(data))
    for i in range(n):
        row = data[i]
        prompt = str(row.get(qfield, "")).strip()
        if not prompt:
            continue
        case_id = str(row.get(id_field)) if id_field else f"case-{i:04d}"
        # metadata: query dışındaki tüm alanlar (ticker/category/difficulty vs.)
        meta = {k: row[k] for k in columns if k != qfield}
        cases.append(Case(index=i, case_id=case_id, prompt=prompt, metadata=meta))
    return cases


# --- Yardımcılar ------------------------------------------------------------


def _ensure_utf8_stdout() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        except Exception:
            pass


def _one_line(value: Any, limit: int = 200) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _slugify(name: str) -> str:
    name = re.sub(r"\s+", "_", name.strip().lower())
    return re.sub(r"[^a-z0-9_\-]", "", name).strip("_-")


def serialize_trace(trace: list) -> list[dict[str, Any]]:
    """ReAct izini (adım adım Thought / reasoning / Action / Observation)
    okunası özet JSON'una çevirir — ajanın nasıl akıl yürüttüğünü görebilmek için."""
    steps: list[dict[str, Any]] = []
    for i, s in enumerate(trace, 1):
        steps.append({
            "step": i,
            "thought": s.thought,
            "reasoning": s.reasoning,          # modelin düşünme çıktısı (varsa)
            "action": s.action,
            "action_input": s.action_input,
            "observation": s.observation,
        })
    return steps


def ask_test_name(fallback_prefix: str = "run") -> str:
    """Koşu adını terminalden sorar; boş girilirse zaman damgası kullanır."""
    fallback = time.strftime(f"{fallback_prefix}_%Y%m%d_%H%M%S")
    try:
        raw = input(f"Bu test koşusu için bir isim gir [{fallback}]: ")
    except EOFError:
        raw = ""
    return _slugify(raw) or fallback


# --- Koşucu -----------------------------------------------------------------


class EvalRunner:
    def __init__(self, version: str, output_dir: Path, model: str | None,
                 max_steps: int, temperature: float, is_synthetic: bool,
                 framework: str, progress: bool = True):
        self.version = version
        self.output_dir = output_dir
        self.model = model
        self.max_steps = max_steps
        self.temperature = temperature
        self.is_synthetic = is_synthetic
        self.framework = framework
        self.progress = progress

        self.summaries: list[dict[str, Any]] = []   # okunası özet
        self.schemas: list[dict[str, Any]] = []      # a.json'a uygun RunResult'lar
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.output_dir / f"progress_{version}.jsonl"
        if progress:
            self.progress_path.write_text("", encoding="utf-8")

    def _event(self, obj: dict[str, Any]) -> None:
        if not self.progress:
            return
        with self.progress_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), **obj}, ensure_ascii=False) + "\n")

    def run_case(self, case: Case, pos: int, total: int) -> None:
        from react_agent import ReActAgent
        from react_agent.llm import HuggingFaceLLM
        from react_agent.run_schema import to_run_result_schema

        print(f"\n[{pos}/{total}] case_id={case.case_id}")
        print(f"  PROMPT: {_one_line(case.prompt, 160)}")
        self._event({"event": "case_start", "pos": pos, "total": total,
                     "case_id": case.case_id, "prompt": case.prompt})

        llm = HuggingFaceLLM(model=self.model, temperature=self.temperature,
                             max_tokens=4096)
        model_name = llm.model
        model_params = {"temperature": llm.temperature, "top_p": None,
                        "seed": None, "max_tokens": llm.max_tokens}
        agent = ReActAgent(llm=llm, max_steps=self.max_steps, verbose=False)

        t0 = time.time()
        try:
            rr = agent.run(case.prompt)
            schema = to_run_result_schema(
                rr, prompt=case.prompt, model=model_name, model_params=model_params,
                case_id=case.case_id, framework=self.framework,
                is_synthetic=self.is_synthetic, case_metadata=case.metadata or None,
            )
            summary = {
                "case_id": case.case_id, "prompt": case.prompt,
                "success": rr.success, "status": rr.status, "answer": rr.answer,
                "steps": rr.steps, "tool_calls": rr.tool_calls,
                "tools_used": rr.tools_used, "total_tokens": rr.total_tokens,
                "duration_ms": int(rr.elapsed_seconds * 1000), "error": None,
                "trace": serialize_trace(rr.trace),   # reasoning adımları
                "scratchpad": rr.scratchpad,          # modele beslenen tam transcript
            }
            print(f"  -> {rr.status} | adım={rr.steps} araç={rr.tool_calls} "
                  f"({', '.join(rr.tools_used) or '-'}) token={rr.total_tokens} "
                  f"süre={rr.elapsed_seconds:.1f}sn")
            print(f"  CEVAP: {_one_line(rr.answer, 220)}")
        except Exception as exc:  # ajan/ağ hatası: koşuyu düşürme, hatayı kaydet
            err = f"{type(exc).__name__}: {exc}"
            print(f"  -> HATA: {err}", file=sys.stderr)
            traceback.print_exc()
            schema = self._error_schema(case, model_name, model_params,
                                        err, time.time() - t0)
            summary = {
                "case_id": case.case_id, "prompt": case.prompt, "success": False,
                "status": "error", "answer": None, "steps": 0, "tool_calls": 0,
                "tools_used": [], "total_tokens": 0,
                "duration_ms": int((time.time() - t0) * 1000), "error": err,
                "trace": [], "scratchpad": "",
            }

        self.summaries.append(summary)
        self.schemas.append(schema)
        self._event({"event": "case_finish", "pos": pos, **summary})
        self.save()  # her case sonrası kaydet (uzun koşuda ilerleme kaybolmasın)

    def _error_schema(self, case, model_name, model_params, err, elapsed):
        from react_agent.agent import RunResult
        from react_agent.run_schema import to_run_result_schema
        rr = RunResult(answer=None, success=False, status="error",
                       elapsed_seconds=elapsed)
        return to_run_result_schema(
            rr, prompt=case.prompt, model=model_name, model_params=model_params,
            case_id=case.case_id, framework=self.framework,
            is_synthetic=self.is_synthetic, case_metadata=case.metadata or None,
            error_info={"error_type": "run_exception", "error_message": err,
                        "failed_step_id": None},
        )

    def save(self) -> tuple[Path, Path]:
        # 1) Okunası özet + toplam istatistik
        n = len(self.summaries)
        ok = sum(1 for s in self.summaries if s["success"])
        summary_payload = {
            "test_name": self.version,          # koşunun adı (en başta)
            "model": self.model or "(env HF_MODEL)",
            "dataset": DATASET_NAME, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": n, "succeeded": ok,
            "avg_steps": round(sum(s["steps"] for s in self.summaries) / n, 2) if n else 0,
            "avg_tokens": round(sum(s["total_tokens"] for s in self.summaries) / n, 1) if n else 0,
            "avg_duration_ms": round(sum(s["duration_ms"] for s in self.summaries) / n, 1) if n else 0,
            "results": self.summaries,
        }
        summary_path = self.output_dir / f"results_{self.version}.json"
        summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False),
                                encoding="utf-8")

        # 2) a.json şemasına uygun RunResult dizisi (asıl eval çıktısı)
        schema_path = self.output_dir / f"results_{self.version}_schema.json"
        schema_path.write_text(json.dumps(self.schemas, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        return summary_path, schema_path


# --- Şema doğrulama (opsiyonel) ---------------------------------------------


def validate_against_schema(schema_objects: list[dict]) -> bool:
    """Üretilen her RunResult'ı test/a.json şemasıyla doğrular."""
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    all_ok = True
    for i, obj in enumerate(schema_objects):
        errors = sorted(validator.iter_errors(obj), key=lambda e: e.path)
        if errors:
            all_ok = False
            print(f"\n[ŞEMA HATASI] #{i} (case_id={obj.get('case_id')}):", file=sys.stderr)
            for e in errors[:5]:
                loc = "/".join(str(p) for p in e.path)
                print(f"  - {loc or '(kök)'}: {e.message}", file=sys.stderr)
    if all_ok:
        print(f"\n✅ Şema doğrulaması geçti: {len(schema_objects)} kayıt a.json'a uygun.")
    else:
        print("\n❌ Bazı kayıtlar şemaya uymuyor (yukarıda).", file=sys.stderr)
    return all_ok


# --- CLI --------------------------------------------------------------------


def main() -> int:
    _ensure_utf8_stdout()
    p = argparse.ArgumentParser(description="Dataset query'leriyle canlı ReAct eval koşucusu.")
    p.add_argument("--dataset", default=DATASET_NAME, help="HF dataset adı.")
    p.add_argument("--split", default=None, help="Dataset split'i (varsayılan: ilk mevcut).")
    p.add_argument("--query-field", default=None, help="Query metninin alan adı (otomatik bulunur).")
    p.add_argument("--limit", type=int, default=None, help="İlk N query'yi çalıştır.")
    p.add_argument("--index", type=int, nargs="+", help="Sadece bu index'lerdeki query'ler.")
    p.add_argument("--list", action="store_true", help="Query'leri listele, çalıştırma.")
    p.add_argument("--model", default=None, help="Model (varsayılan: env HF_MODEL).")
    p.add_argument("--max-steps", type=int, default=10, help="Ajan adım sınırı.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--framework", default="react-scratch", help="Şemadaki framework etiketi.")
    p.add_argument("--synthetic", action="store_true", help="is_synthetic=true olarak işaretle.")
    p.add_argument("--version", default=None, help="Çıktı dosya etiketi (varsayılan: zaman damgası).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--no-progress", action="store_true", help="JSONL ilerleme akışını kapat.")
    p.add_argument("--validate", action="store_true", help="Çıktıyı a.json şemasıyla doğrula.")
    args = p.parse_args()

    import os
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")):
        print("HATA: HF_TOKEN tanımlı değil (.env'e ekle).", file=sys.stderr)
        return 1

    print(f"Dataset yükleniyor: {args.dataset} ...")
    cases = load_cases(args.dataset, args.split, args.query_field, args.limit)
    if args.index:
        wanted = set(args.index)
        cases = [c for c in cases if c.index in wanted]
    if not cases:
        print("Çalıştırılacak query bulunamadı.", file=sys.stderr)
        return 1

    if args.list:
        print(f"\n{len(cases)} query:")
        for c in cases:
            print(f"  [{c.index}] {c.case_id}: {_one_line(c.prompt, 120)}")
        return 0

    # Test adı: --version verilmişse onu, yoksa koşu başında terminalden sor.
    version = _slugify(args.version) if args.version else ask_test_name()
    runner = EvalRunner(
        version=version, output_dir=args.output_dir, model=args.model,
        max_steps=args.max_steps, temperature=args.temperature,
        is_synthetic=args.synthetic, framework=args.framework,
        progress=not args.no_progress,
    )
    print(f"\nKoşu '{version}' — {len(cases)} query (model={args.model or 'env HF_MODEL'}, "
          f"max_steps={args.max_steps})")

    try:
        for pos, case in enumerate(cases, 1):
            runner.run_case(case, pos, len(cases))
    except KeyboardInterrupt:
        print("\nKullanıcı durdurdu; şu ana kadarki sonuçlar kaydediliyor…", file=sys.stderr)

    summary_path, schema_path = runner.save()
    ok = sum(1 for s in runner.summaries if s["success"])
    print(f"\n{'='*60}")
    print(f"Bitti: {len(runner.summaries)} query, {ok} başarılı.")
    print(f"Özet:  {summary_path}")
    print(f"Şema:  {schema_path}")

    rc = 0
    if args.validate:
        if not validate_against_schema(runner.schemas):
            rc = 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
