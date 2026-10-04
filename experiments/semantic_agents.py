"""A semantic (non-numeric) variant of the real Planner->Worker->Verifier system,
for Experiment 5. The task set is BIG-Bench Hard ``tracking_shuffled_objects_three_objects``
(public; gold answers are option letters, i.e. categories; solving it needs a
multi-step chain of *semantic* state updates -- who holds which object after
each swap -- with no comparable number anywhere in the trace).

The Worker's typed actions mirror real_agents.py but replace the numeric CALC
tool with a free-text NOTE step (one state update per turn):

    NOTE: <one reasoning step, e.g. the state after one swap>
    ANSWER: <option letter>

Success = the ANSWER letter matches the gold option. Nothing is injected; failures
are natural. Populations are collected exactly as in Experiment 4A.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from refalign.types import Step, Trace

PLANNER_SYS = ("You are Planner. Give a one-sentence plan to solve the reasoning task. Be brief.")
WORKER_SYS = (
    "You are Worker, solving a reasoning task step by step. Each turn output exactly one action:\n"
    "  NOTE: <one reasoning step, e.g. the state after one event>\n"
    "  ANSWER: <option letter, e.g. (B)>\n"
    "Use NOTE for every intermediate step. When confident, output ANSWER. One action per turn.")
VERIFIER_SYS = ("You are Verifier. Given the worker's answer, reply exactly:\n"
                "VERDICT: CORRECT or VERDICT: WRONG\nFINAL: <option letter>")

_OPT = re.compile(r"\(([A-Z])\)")


def parse_option(s: str) -> Optional[str]:
    """First option letter in '(X)' form; falls back to a bare capital letter token."""
    m = _OPT.search(s or "")
    if m:
        return m.group(1)
    m = re.search(r"\b([A-E])\b", s or "")
    return m.group(1) if m else None


def _parse_action(raw: str) -> Tuple[str, str]:
    """Typed-action parser. Unlike real_agents._parse_action it also accepts the
    keyword without a colon at the start of a line ('NOTE after swap 1 ...'),
    because in a first pass 57% of failures were NOTE lines mis-read as ANSWER
    for want of a colon (kept as results/new/logs/semantic_variant_v1_strictparser.json)."""
    for key in ("ANSWER", "NOTE"):
        m = re.search(rf"{key}\s*[:=]\s*(.+)", raw, re.I | re.S)
        if m:
            return key, m.group(1).strip().splitlines()[0].strip()
    for line in raw.strip().splitlines():
        m = re.match(r"^\W*(ANSWER|NOTE)\b\W*(.*)$", line.strip(), re.I)
        if m and m.group(2).strip():
            return m.group(1).upper(), m.group(2).strip()
    return "ANSWER", raw.strip()[:60]


def run_once_semantic(llm, task: str, answer: str, temperature: float = 1.0,
                      max_worker_turns: int = 8) -> Tuple[Trace, bool]:
    steps: List[Step] = []
    plan = llm.chat(PLANNER_SYS, f"Task: {task}")
    steps.append(Step(len(steps), "Planner", "plan", plan))
    history = f"Task: {task}\nPlan: {plan}\n"
    final = None
    for _ in range(max_worker_turns):
        raw = llm.chat(WORKER_SYS, history + "\nYour next action:")
        act, arg = _parse_action(raw)
        if act == "NOTE":
            history += f"\nWorker NOTE {arg}"
            steps.append(Step(len(steps), "Worker", "note", f"NOTE {arg}"))
        else:
            final = parse_option(arg)
            steps.append(Step(len(steps), "Worker", "answer", f"ANSWER {arg}"))
            break
    verdict = llm.chat(VERIFIER_SYS, f"Task: {task}\nWorker's answer: {final}")
    steps.append(Step(len(steps), "Verifier", "check", verdict))
    success = final is not None and final == parse_option(answer)
    return Trace("", "", steps, success, query=task, final_answer=str(final)), success
