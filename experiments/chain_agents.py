"""Chained-computation multi-agent traces with controllable length and clean
step-level ground truth — the setting where single-trace localisation is hard
(a long dependent chain) but a successful sibling makes the divergence obvious.

System: Planner (LLM) -> Worker (a calculator *tool*, one op per step) ->
Verifier (LLM). The chain of ops is fixed per task, so all healthy runs compute
the same running values; a content fault injected at op i makes the failing run's
value diverge at exactly step i and propagate. Worker steps use a deterministic
tool (as real agents do), so the traces are real (LLM plan + tool trace + LLM
verify) and cheap (2 LLM calls per run).
"""
from __future__ import annotations

import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from refalign.types import Step, Trace, Annotation, RunPopulation


def _det_seed(*parts) -> int:
    """Deterministic 32-bit seed from parts (unlike Python's salted hash())."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF

_OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}


def make_chain(rng: np.random.RandomState, length: int) -> Tuple[int, List[Tuple[str, int]], List[float]]:
    start = int(rng.randint(3, 15))
    ops: List[Tuple[str, int]] = []
    running = [float(start)]
    for _ in range(length):
        op = str(rng.choice(["+", "*", "-", "+"]))       # bias away from too many '*'
        operand = int(rng.randint(2, 9))
        val = _OPS[op](running[-1], operand)
        # keep magnitudes reasonable
        if abs(val) > 5000:
            op, operand = "+", int(rng.randint(2, 9))
            val = running[-1] + operand
        ops.append((op, operand))
        running.append(float(val))
    return start, ops, running


def _describe(start: int, ops: List[Tuple[str, int]]) -> str:
    parts = [f"start at {start}"]
    for op, b in ops:
        parts.append({"+": f"add {b}", "-": f"subtract {b}", "*": f"multiply by {b}"}[op])
    return "Compute step by step: " + ", then ".join(parts) + ". Give the final number."


_FORMATS = ["CALC: {a} {op} {b} = {v}", "Compute {a} {op} {b} -> {v}",
            "step: {a}{op}{b}={v}", "applying {op}{b} to {a} gives {v}"]


def run_chain(llm, start: int, ops: List[Tuple[str, int]], running: List[float],
              inject: Optional[Tuple[int, str]] = None,
              rng: Optional[np.random.RandomState] = None
              ) -> Tuple[Trace, bool, Optional[int], Optional[str]]:
    """Run one chain. ``inject=(op_index, kind)`` corrupts op ``op_index``.

    If ``rng`` is given, benign variation is added to *this* run: the calc-content
    wording is randomised and a harmless 'note' step is occasionally inserted, so
    successful siblings are NOT identical and the alignment must reconcile indels.
    """
    task = _describe(start, ops)
    steps: List[Step] = []
    fault_step = fault_agent = None

    plan = llm.chat("You are Planner. In one sentence, state the order of operations to do. Be brief.",
                    f"Task: {task}") if llm is not None else "Apply each operation in order."
    steps.append(Step(len(steps), "Planner", "plan", plan))

    cur = float(start)
    for i, (op, b) in enumerate(ops):
        true_val = _OPS[op](cur, b)
        val = true_val
        if inject is not None and i == inject[0]:
            fault_step = len(steps)
            fault_agent = "Worker"
            if inject[1] == "bad_calc":            # content fault: wrong value
                val = true_val + 3
            elif inject[1] == "wrong_op":          # content fault: wrong operation
                val = _OPS["+"](cur, b) if op != "+" else _OPS["*"](cur, b)
        fmt = _FORMATS[0] if rng is None else str(rng.choice(_FORMATS))
        steps.append(Step(len(steps), "Worker", "calc", fmt.format(a=cur, op=op, b=b, v=val)))
        # benign 'note' step (an indel) that carries no decisive value
        if rng is not None and rng.rand() < 0.2:
            steps.append(Step(len(steps), "Worker", "note", f"(so far the total is {val})"))
        cur = val                                  # propagate (wrong value flows on)

    steps.append(Step(len(steps), "Worker", "answer", f"ANSWER: {cur}"))
    verdict = llm.chat("You are Verifier. Reply 'VERDICT: CORRECT' or 'VERDICT: WRONG' then 'FINAL: <number>'.",
                       f"Task: {task}\nWorker's final answer: {cur}") if llm is not None else "VERDICT: CORRECT"
    steps.append(Step(len(steps), "Verifier", "check", verdict))

    true_answer = running[-1]
    success = abs(cur - true_answer) < 1e-6
    trace = Trace(run_id="", query_id="", steps=steps, success=success,
                  query=task, final_answer=str(cur))
    return trace, success, fault_step, fault_agent


def build_population(llm, task_idx: int, length: int, seed: int,
                     n_clean: int = 4, n_fault: int = 4, noise: bool = False
                     ) -> Tuple[RunPopulation, Dict[str, Annotation]]:
    rng = np.random.RandomState(1000 * seed + task_idx)
    start, ops, running = make_chain(rng, length)
    qid = f"L{length}_t{task_idx:02d}"
    runs: List[Trace] = []
    anns: Dict[str, Annotation] = {}

    def _rng(tag):    # deterministic per-run RNG so noisy siblings differ (reproducible)
        return np.random.RandomState(_det_seed(seed, task_idx, tag)) if noise else None

    for r in range(n_clean):
        tr, ok, _, _ = run_chain(llm, start, ops, running, rng=_rng(f"c{r}"))
        tr.run_id, tr.query_id = f"{qid}_c{r:02d}", qid
        runs.append(tr)

    kinds = ["bad_calc", "wrong_op"]
    for r in range(n_fault):
        op_i = int(rng.randint(1, length))          # interior op
        kind = kinds[r % len(kinds)]
        tr, ok, fstep, fagent = run_chain(llm, start, ops, running, inject=(op_i, kind),
                                          rng=_rng(f"f{r}"))
        rid = f"{qid}_f{r:02d}"
        tr.run_id, tr.query_id, tr.success = rid, qid, False
        runs.append(tr)
        if fstep is not None:
            anns[rid] = Annotation(qid, rid, fagent, fstep, reason=f"chain {kind} at op {op_i}",
                                   meta={"fault_type": kind, "length": length})
    return RunPopulation(qid, runs, _describe(start, ops)), anns
