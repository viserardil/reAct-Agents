

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _iso(value):
    """date-time alanı asla None olmasın (şema string zorunlu kılıyor)."""
    return value or _now()


def _step_type(step, is_last, success):
    """TraceStep -> şemadaki step_type (planning|tool_call|synthesis|reflection|...)."""
    if step.action:
        return "tool_call"
    if is_last and success:
        return "synthesis"        # nihai cevabı üreten adım
    return "reflection"           # araçsız ara adım (ör. format düzeltme)


def _tool_calls(step):
    """Bir adımdaki araç çağrısını şemadaki ToolCall listesine çevirir."""
    if not step.action:
        return []
    return [{
        "tool_call_id": f"tc-{uuid.uuid4()}",
        "tool_name": step.action,
        "tool_input": {"input": step.action_input or ""},
        "tool_output": step.observation,
        "success": bool(step.tool_success),
        "error_message": None if step.tool_success else step.observation,
        "latency_ms": round(step.tool_ms, 3),
    }]


def _llm_calls(step):
    """Her adım tek bir LLM çağrısı içerir (bizim döngümüzde)."""
    return [{
        "llm_call_id": f"llm-{uuid.uuid4()}",
        "input_tokens": step.input_tokens,
        "output_tokens": step.output_tokens,
        "duration_ms": round(step.llm_ms, 3),
    }]


def _steps(trace, success):
    steps = []
    for i, st in enumerate(trace):
        is_last = i == len(trace) - 1
        steps.append({
            "step_id": f"step-{uuid.uuid4()}",
            "step_index": i,                       # 0-indexed
            "step_type": _step_type(st, is_last, success),
            "started_at": _iso(st.started_at),
            "ended_at": _iso(st.ended_at),
            "duration_ms": round(st.llm_ms + st.tool_ms, 3),
            # Modelin ham düşünme çıktısı (reasoning_content); yoksa Thought.
            "reasoning_content": st.reasoning or st.thought,
            "input_context": None,
            "output": st.thought,
            "tool_calls": _tool_calls(st),
            "llm_calls": _llm_calls(st),
        })
    return steps


def to_run_result_schema(
    run_result,
    *,
    prompt,
    model,
    model_params,
    case_id,
    framework="react-scratch",
    repetition_index=1,
    is_synthetic=False,
    case_metadata=None,
    run_id=None,
    error_info=None,
):
    """RunResult'ı v2.0.0 RunResult şemasına uyan bir dict'e çevirir.

    prompt        : agent'a verilen tam soru
    model         : koşumun modeli (ör. HF_MODEL)
    model_params  : {"temperature": float, "max_tokens": int, ...}
    case_id       : görev/senaryo kimliği (gruplama anahtarı)
    framework     : framework adı
    error_info    : status=error ise {"error_type","error_message","failed_step_id"}
    """
    steps = _steps(run_result.trace, run_result.success)

    agent = {
        "agent_id": "react-agent-0",
        "role": "react-agent",
        "system_prompt_hash": None,
        "steps": steps,
        "tokens": {
            "input": run_result.input_tokens,
            "output": run_result.output_tokens,
        },
    }

    status = "error" if error_info else run_result.status

    return {
        "schema_version": "2.0.0",
        "run_id": run_id or str(uuid.uuid4()),
        "framework": framework,
        "repetition_index": repetition_index,
        "case_id": str(case_id),
        "case_metadata": case_metadata,
        "is_synthetic": is_synthetic,
        "prompt": prompt,
        "model": model,
        "model_params": model_params,
        "num_agents": 1,
        "total_steps": len(steps),
        "agents": [agent],
        "final_answer": run_result.answer,
        "status": status,
        "error_info": error_info,
        "tokens": {
            "input": run_result.input_tokens,
            "output": run_result.output_tokens,
            "total": run_result.total_tokens,
        },
        "latency": {
            "total_ms": round(run_result.elapsed_seconds * 1000, 3),
            "planning_ms": None,
            "tool_execution_ms": round(sum(s.tool_ms for s in run_result.trace), 3),
            "llm_inference_ms": round(sum(s.llm_ms for s in run_result.trace), 3),
        },
        "timestamps": {
            "started_at": _iso(run_result.started_at),
            "ended_at": _iso(run_result.ended_at),
        },
        "trace_id": None,
        # Ortak şemaya sığmayan ekstra: modele beslenen tam çalışma izi.
        "framework_specific": {"scratchpad": getattr(run_result, "scratchpad", "")},
    }
