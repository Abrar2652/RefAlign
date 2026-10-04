"""Richer benchmarks that close the evaluation gaps of the headline chain corpus:

  * MULTI-CULPRIT: the chain's calc steps rotate through several worker agents
    (round-robin), so the responsible agent varies by step and agent-level
    accuracy is a real test (chance ~ 1/#agents), not trivially the one busy agent.
  * FAULT-TYPE DIVERSITY: value / wrong-operation / skipped-step / repeated-step
    faults, each with ground-truth (agent, step), reported per type.
  * SOLUTION DIVERSITY: a reorderable-sum variant where *successful* runs
    legitimately differ (different valid orders -> different intermediate values),
    stressing the "divergence from a sibling" signal and testing content-aware
    sibling selection.

Corpus is deterministic (calculator-tool workers) so it is free and reproducible;
the LLM is spent only on the attribution methods being compared.
"""
from __future__ import annotations

import zlib
from typing import Dict, List, Tuple

import numpy as np

from refalign.types import Step, Trace, Annotation, RunPopulation


def _det_seed(*parts) -> int:
    """Deterministic 32-bit seed (Python's hash() is salted per process)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF

_OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}
WORKER_AGENTS = ["Analyst", "Coder", "Solver", "Checker"]
FAULT_TYPES = ["value", "wrong_op", "skip", "repeat"]
_FMTS = ["CALC: {a} {op} {b} = {v}", "compute {a}{op}{b} -> {v}", "{a} {op} {b} = {v}"]


def _chain(rng: np.random.RandomState, length: int):
    start = int(rng.randint(3, 15))
    ops, running = [], [float(start)]
    for _ in range(length):
        op = str(rng.choice(["+", "*", "-", "+"]))
        b = int(rng.randint(2, 9))
        v = _OPS[op](running[-1], b)
        if abs(v) > 5000:
            op, b = "+", int(rng.randint(2, 9)); v = running[-1] + b
        ops.append((op, b)); running.append(float(v))
    return start, ops, running


# ---------------------------------------------------------------------------
# Multi-culprit chain with 4 fault types
# ---------------------------------------------------------------------------
def run_multiculprit(start, ops, running, agents, inject=None, rng=None):
    steps: List[Step] = [Step(0, "Planner", "plan", "apply the operations in order")]
    cur = float(start)
    fstep = fagent = ftype = None
    for i, (op, b) in enumerate(ops):
        ag = agents[i % len(agents)]
        true = _OPS[op](cur, b)
        val = true
        emit = True
        if inject is not None and i == inject[0]:
            kind = inject[1]; fagent = ag; ftype = kind
            if kind == "value":
                val = true + 3
            elif kind == "wrong_op":
                val = _OPS["+"](cur, b) if op != "+" else _OPS["*"](cur, b)
            elif kind == "repeat":
                val = cur                      # no progress: repeats prior total
            elif kind == "skip":
                emit = False
                fstep = len(steps)             # index the missing calc would hold
        if emit:
            if inject is not None and i == inject[0]:
                fstep = len(steps)
            fmt = _FMTS[0] if rng is None else str(rng.choice(_FMTS))
            steps.append(Step(len(steps), ag, "calc", fmt.format(a=cur, op=op, b=b, v=val)))
            if rng is not None and rng.rand() < 0.2:
                steps.append(Step(len(steps), ag, "note", f"(running total {val})"))
        cur = val
    steps.append(Step(len(steps), "Solver", "answer", f"ANSWER: {cur}"))
    steps.append(Step(len(steps), "Verifier", "check", "VERDICT: CORRECT"))
    success = abs(cur - running[-1]) < 1e-6
    return (Trace("", "", steps, success, query="multi-op chain",
                  final_answer=str(cur)), success, fstep, fagent, ftype)


def build_multiculprit(task_idx: int, length: int, seed: int, n_clean: int = 5,
                       n_fault: int = 8, noise: bool = True, n_agents: int = 4
                       ) -> Tuple[RunPopulation, Dict[str, Annotation]]:
    rng = np.random.RandomState(1000 * seed + task_idx)
    start, ops, running = _chain(rng, length)
    agents = WORKER_AGENTS[:n_agents]
    qid = f"mc_L{length}_t{task_idx:02d}"
    runs, anns = [], {}

    def _rng(tag):
        return np.random.RandomState(_det_seed(seed, task_idx, tag)) if noise else None

    for r in range(n_clean):
        tr, *_ = run_multiculprit(start, ops, running, agents, rng=_rng(f"c{r}"))
        tr.run_id, tr.query_id = f"{qid}_c{r:02d}", qid
        runs.append(tr)
    for r in range(n_fault):
        op_i = int(rng.randint(1, length))
        kind = FAULT_TYPES[r % len(FAULT_TYPES)]
        tr, ok, fstep, fagent, ftype = run_multiculprit(
            start, ops, running, agents, inject=(op_i, kind), rng=_rng(f"f{r}"))
        tr.run_id, tr.query_id, tr.success = f"{qid}_f{r:02d}", qid, False
        runs.append(tr)
        if fstep is not None:
            anns[tr.run_id] = Annotation(qid, tr.run_id, fagent, fstep,
                                         reason=f"{ftype} at op {op_i}",
                                         meta={"fault_type": ftype})
    return RunPopulation(qid, runs, "multi-op chain"), anns


# ---------------------------------------------------------------------------
# Solution-diversity (reorderable sum): successful runs legitimately differ
# ---------------------------------------------------------------------------
def run_sum(numbers, agents, order, inject=None, rng=None):
    steps: List[Step] = [Step(0, "Planner", "plan", "add the numbers")]
    cur = 0.0
    fstep = fagent = ftype = None
    for pos, idx in enumerate(order):
        ag = agents[pos % len(agents)]
        b = numbers[idx]
        true = cur + b
        val = true
        if inject is not None and pos == inject[0]:
            fagent = ag; ftype = inject[1]; fstep = len(steps)
            val = true + 3
        steps.append(Step(len(steps), ag, "calc", f"add {b}: {cur}+{b}={val}"))
        cur = val
    steps.append(Step(len(steps), "Solver", "answer", f"ANSWER: {cur}"))
    steps.append(Step(len(steps), "Verifier", "check", "VERDICT: CORRECT"))
    success = abs(cur - sum(numbers)) < 1e-6
    return (Trace("", "", steps, success, query="reorderable sum",
                  final_answer=str(cur)), success, fstep, fagent, ftype)


def build_diverse(task_idx: int, k: int, seed: int, n_clean: int = 6, n_fault: int = 6,
                  n_strategies: int = 3, n_agents: int = 4
                  ) -> Tuple[RunPopulation, Dict[str, Annotation]]:
    """Reorderable-sum population: successful runs use one of a few valid orders,
    so siblings are genuinely diverse (different partial sums, same total)."""
    rng = np.random.RandomState(1000 * seed + task_idx)
    numbers = [int(rng.randint(2, 20)) for _ in range(k)]
    agents = WORKER_AGENTS[:n_agents]
    # a small set of valid strategies (orders): left-to-right, reversed, sorted
    strategies = [list(range(k)), list(range(k))[::-1],
                  sorted(range(k), key=lambda i: numbers[i])]
    strategies = strategies[:n_strategies]
    qid = f"div_k{k}_t{task_idx:02d}"
    runs, anns = [], {}

    for r in range(n_clean):
        order = strategies[r % len(strategies)]
        tr, *_ = run_sum(numbers, agents, order)
        tr.run_id, tr.query_id = f"{qid}_c{r:02d}", qid
        runs.append(tr)
    for r in range(n_fault):
        order = strategies[r % len(strategies)]
        pos = int(rng.randint(1, k))
        tr, ok, fstep, fagent, ftype = run_sum(numbers, agents, order, inject=(pos, "value"))
        tr.run_id, tr.query_id, tr.success = f"{qid}_f{r:02d}", qid, False
        runs.append(tr)
        if fstep is not None:
            anns[tr.run_id] = Annotation(qid, tr.run_id, fagent, fstep,
                                         reason=f"value at pos {pos}",
                                         meta={"fault_type": "value"})
    return RunPopulation(qid, runs, "reorderable sum"), anns
