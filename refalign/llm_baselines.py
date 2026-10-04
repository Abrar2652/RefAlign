"""The Who&When LLM attribution baselines (Zhang et al., ICML 2025, Table 1):
All-at-Once, Step-by-Step, and Binary-Search. Each consumes a single failing
trace and an LLM (:mod:`refalign.llm`) and returns a :class:`Prediction`.

These operate on one trace at a time, so — unlike RefAlign — they run directly
on raw Who&When.
"""
from __future__ import annotations

import re
from difflib import get_close_matches
from typing import List, Optional, Tuple

from .types import Trace, Prediction

_MAX_CHARS = 700   # truncate each step's content to keep prompts bounded


def _fmt_step(s, max_chars=_MAX_CHARS) -> str:
    c = (s.content or s.action or "").replace("\n", " ").strip()
    if len(c) > max_chars:
        c = c[:max_chars] + " …"
    return f"[{s.index}] {s.agent}: {c}"


def _transcript(trace: Trace, lo: int = 0, hi: Optional[int] = None) -> str:
    hi = len(trace) if hi is None else hi
    return "\n".join(_fmt_step(s) for s in trace.steps[lo:hi])


def _roster(trace: Trace) -> List[str]:
    seen: List[str] = []
    for s in trace.steps:
        if s.agent not in seen:
            seen.append(s.agent)
    return seen


def _parse(raw: str, trace: Trace) -> Tuple[str, int]:
    roster = _roster(trace)
    agent = roster[-1]
    step = len(trace) - 1
    m = re.search(r"AGENT\s*[:=]\s*([^\n|]+)", raw, re.I)
    if m:
        cand = m.group(1).strip().strip(".` ")
        if cand in roster:
            agent = cand
        else:
            close = get_close_matches(cand, roster, n=1, cutoff=0.6)
            agent = close[0] if close else agent
    m = re.search(r"STEP\s*[:=]\s*(\d+)", raw, re.I)
    if m:
        step = max(0, min(int(m.group(1)), len(trace) - 1))
    return agent, step


_SYS = ("You are an expert at debugging multi-agent LLM systems. You will be "
        "shown a failed run and must identify the single decisive error: the "
        "earliest (agent, step) such that correcting that action would have made "
        "the run succeed. Answer on one line exactly as: AGENT: <name> | STEP: <index>.")


def all_at_once(trace: Trace, llm) -> Prediction:
    roster = _roster(trace)
    user = (f"Task: {trace.query}\n\nAgents: {', '.join(roster)}\n\n"
            f"Failed run (step index in brackets):\n{_transcript(trace)}\n\n"
            "Which agent and step is the decisive error?")
    raw = llm.chat(_SYS, user)
    agent, step = _parse(raw, trace)
    return Prediction(trace.run_id, agent, step, 0.5,
                      meta={"baseline": "all_at_once", "raw": raw[:200]})


def binary_search(trace: Trace, llm) -> Prediction:
    """Divide-and-conquer: localize the half containing the decisive error, recurse."""
    lo, hi = 0, len(trace)
    while hi - lo > 2:
        mid = (lo + hi) // 2
        user = (f"Task: {trace.query}\n\nThe decisive error is somewhere in steps "
                f"[{lo}..{hi-1}] of this failed run.\n\nFirst half (steps {lo}..{mid-1}):\n"
                f"{_transcript(trace, lo, mid)}\n\nSecond half (steps {mid}..{hi-1}):\n"
                f"{_transcript(trace, mid, hi)}\n\n"
                "Does the decisive error occur in the FIRST or SECOND half? "
                "Answer exactly FIRST or SECOND.")
        raw = llm.chat(_SYS, user).strip().upper()
        if "SECOND" in raw:
            lo = mid
        else:
            hi = mid
    user = (f"Task: {trace.query}\n\nThe decisive error is in steps [{lo}..{hi-1}]:\n"
            f"{_transcript(trace, lo, hi)}\n\nGive the exact agent and step.")
    raw = llm.chat(_SYS, user)
    agent, step = _parse(raw, trace)
    step = max(lo, min(step, hi - 1))
    return Prediction(trace.run_id, agent, step, 0.5,
                      meta={"baseline": "binary_search"})


def step_by_step(trace: Trace, llm, window: int = 4) -> Prediction:
    """Walk forward; stop at the first step judged to be the decisive error."""
    roster = _roster(trace)
    for k in range(len(trace)):
        ctx = _transcript(trace, max(0, k - window), k + 1)
        user = (f"Task: {trace.query}\nAgents: {', '.join(roster)}\n\n"
                f"Recent steps (current is [{k}]):\n{ctx}\n\n"
                f"Is step [{k}] the decisive error that dooms the run? "
                "Answer YES or NO; if YES also give AGENT: <name> | STEP: "
                f"{k}.")
        raw = llm.chat(_SYS, user)
        if re.search(r"\bYES\b", raw, re.I):
            agent, _ = _parse(raw, trace)
            if agent not in roster:
                agent = trace.agent_at(k)
            return Prediction(trace.run_id, agent, k, 0.5,
                              meta={"baseline": "step_by_step"})
    last = len(trace) - 1
    return Prediction(trace.run_id, trace.agent_at(last), last, 0.0,
                      meta={"baseline": "step_by_step", "fallback": "no_error_found"})


LLM_METHODS = {
    "all_at_once": all_at_once,
    "binary_search": binary_search,
    "step_by_step": step_by_step,
}


def run_llm_baseline(method: str, traces: List[Trace], llm) -> dict:
    fn = LLM_METHODS[method]
    return {t.run_id: fn(t, llm) for t in traces}


# ---------------------------------------------------------------------------
# Matched-information judge (the paper follow-up, Experiment 1).
#
# The single-trace judge above sees only the failure. This judge sees the failure
# AND a successful reference run of the same task -- the same information
# RefAlign's alignment consumes -- so the gap between the two isolates the
# value of the alignment algorithm from the value of having the reference.
# Rendering matches ``contrastive_llm``'s ``show()`` helper (120-char steps).
# ---------------------------------------------------------------------------
_MATCHED_SYS = ("You debug multi-agent runs by comparison. You are given a FAILED run "
                "and a SUCCESSFUL run of the same task. Identify the earliest step in "
                "the FAILED run whose correction would make it succeed. Step indices "
                "are zero-based and shown in brackets. "
                "Answer exactly: STEP: <failing step index> | AGENT: <agent name>.")


def _show_matched(tr: Trace, max_chars: int = 120) -> str:
    return "\n".join(f"  [{s.index}] {s.agent}: {(s.content or s.action)[:max_chars]}"
                     for s in tr.steps)


def matched_judge_prompt(failing: Trace, reference: Trace) -> str:
    """User prompt: full failing run + full successful reference (same task)."""
    return (f"Task: {failing.query}\n\nSUCCESSFUL run:\n{_show_matched(reference)}\n\n"
            f"FAILED run:\n{_show_matched(failing)}\n\n"
            "Identify the earliest step in the FAILED run whose correction would make "
            "it succeed. Answer exactly: STEP: <failing step index> | AGENT: <agent name>.")


def parse_judge_answer(raw: str, trace: Trace) -> Tuple[str, int, bool, bool]:
    """Reuse the existing parser and additionally report parseability.

    Returns ``(agent, step, step_parsed, agent_parsed)``. ``step_parsed`` is
    False when no ``STEP: <int>`` token is present; ``agent_parsed`` is False
    when no ``AGENT: <name>`` token is present. Callers score unparseable
    answers as incorrect (they are *not* silently mapped to the last step).
    """
    agent, step = _parse(raw, trace)
    step_parsed = re.search(r"STEP\s*[:=]\s*(\d+)", raw, re.I) is not None
    if not step_parsed:
        # Tolerate a bracketed index ("STEP: [2]"), which mirrors the "[i]" step
        # rendering in the prompt; gpt-4o-2024-08-06 answers this way. The strict
        # form above is what the existing parser accepts; callers can keep both counts.
        m = re.search(r"STEP\s*[:=]\s*\[\s*(\d+)\s*\]", raw, re.I)
        if m:
            step = max(0, min(int(m.group(1)), len(trace) - 1))
            step_parsed = True
    agent_parsed = re.search(r"AGENT\s*[:=]\s*([^\n|]+)", raw, re.I) is not None
    return agent, step, step_parsed, agent_parsed


def strict_step_parsed(raw: str) -> bool:
    """True when the *existing* parser's STEP regex (no brackets) matches."""
    return re.search(r"STEP\s*[:=]\s*(\d+)", raw, re.I) is not None


UNPARSED_AGENT = "<unparsed>"
UNPARSED_STEP = -1


def matched_judge(failing: Trace, reference: Trace, judge) -> Prediction:
    """LLM judge that sees the failing run and one successful reference run.

    Unparseable outputs are returned as ``(UNPARSED_AGENT, UNPARSED_STEP)`` so
    they score incorrect on both metrics and are countable via ``meta``.
    """
    raw = judge.chat(_MATCHED_SYS, matched_judge_prompt(failing, reference))
    agent, step, step_ok, agent_ok = parse_judge_answer(raw, failing)
    if not step_ok:
        agent, step = UNPARSED_AGENT, UNPARSED_STEP
    elif not agent_ok:
        agent = UNPARSED_AGENT
    return Prediction(failing.run_id, agent, step, 0.5,
                      meta={"baseline": "matched_judge", "reference": reference.run_id,
                            "step_parsed": step_ok, "agent_parsed": agent_ok,
                            "strict_step_parsed": strict_step_parsed(raw), "raw": raw[:400]})
