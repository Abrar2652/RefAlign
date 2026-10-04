"""Baseline attributors from the research grounding (Section 5).

Each baseline exposes the same signature as RefAlign:

    fn(pop, failing, rng) -> Prediction

so the evaluation harness can score them uniformly. Included:

  * ``random``            — uniform agent + uniform step (the reported floor);
  * ``first_step``        — always blame the first action;
  * ``last_step``         — always blame the last action;
  * ``most_active_agent`` — blame the agent that acts most, at its median step;
  * ``longest_agent``     — blame the agent with the most *consecutive* turns;
  * ``edit_nn``           — a sequence-only (no phylogeny) attributor: align the
    failing run to its nearest healthy run by raw edit distance and take the
    earliest out-of-repertoire divergence.

LLM baselines (All-at-Once / Step-by-Step / Binary-Search / one-shot judge) need
a model endpoint; :func:`make_llm_baseline` returns an adapter around a
user-supplied ``call_llm`` function so they slot into the same harness when a key
is available. They are intentionally *not* run offline.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List

import numpy as np

from .types import Trace, Prediction, RunPopulation
from .encode import Alphabet, GAP
from .msa import needleman_wunsch

Attributor = Callable[[RunPopulation, Trace, np.random.RandomState], Prediction]


def random_baseline(pop: RunPopulation, failing: Trace,
                    rng: np.random.RandomState) -> Prediction:
    roster = failing.agents or list({s.agent for s in failing.steps})
    agent = str(rng.choice(roster))
    step = int(rng.randint(0, len(failing)))
    return Prediction(failing.run_id, agent, step, 0.0, meta={"baseline": "random"})


def first_step_baseline(pop: RunPopulation, failing: Trace,
                        rng: np.random.RandomState) -> Prediction:
    return Prediction(failing.run_id, failing.agent_at(0), 0, 0.0,
                      meta={"baseline": "first_step"})


def last_step_baseline(pop: RunPopulation, failing: Trace,
                       rng: np.random.RandomState) -> Prediction:
    last = len(failing) - 1
    return Prediction(failing.run_id, failing.agent_at(last), last, 0.0,
                      meta={"baseline": "last_step"})


def most_active_agent_baseline(pop: RunPopulation, failing: Trace,
                               rng: np.random.RandomState) -> Prediction:
    counts = Counter(s.agent for s in failing.steps)
    agent = counts.most_common(1)[0][0]
    idxs = [s.index for s in failing.steps if s.agent == agent]
    step = int(np.median(idxs)) if idxs else 0
    return Prediction(failing.run_id, agent, step, 0.0,
                      meta={"baseline": "most_active_agent"})


def longest_agent_baseline(pop: RunPopulation, failing: Trace,
                           rng: np.random.RandomState) -> Prediction:
    """Blame the agent holding the longest consecutive run of turns."""
    best_agent = failing.agent_at(0)
    best_len = 0
    best_start = 0
    i = 0
    steps = failing.steps
    while i < len(steps):
        j = i
        while j < len(steps) and steps[j].agent == steps[i].agent:
            j += 1
        if j - i > best_len:
            best_len, best_agent, best_start = j - i, steps[i].agent, i
        i = j
    step = best_start + best_len // 2
    return Prediction(failing.run_id, best_agent, step, 0.0,
                      meta={"baseline": "longest_agent"})


def edit_nn_baseline(pop: RunPopulation, failing: Trace,
                     rng: np.random.RandomState) -> Prediction:
    """Sequence-only nearest-neighbour attributor (no MSA, no tree).

    Encodes runs, finds the nearest successful run by raw Needleman-Wunsch cost,
    aligns to it, and reports the earliest position where the failing run uses a
    symbol absent from *any* successful run's aligned position. This isolates the
    value of the phylogeny: it is RefAlign minus MSA population + tree.
    """
    successes = pop.successes
    if not successes:
        last = len(failing) - 1
        return Prediction(failing.run_id, failing.agent_at(last), last, 0.0,
                          meta={"baseline": "edit_nn", "fallback": "no_successes"})

    alpha = Alphabet(mode="agent_action")
    enc: Dict[str, np.ndarray] = {}
    for run in pop.runs:
        enc[run.run_id] = np.asarray(
            [alpha.encode(s.agent, s.action) for s in run.steps], dtype=np.int64)

    # global healthy symbol multiset per NOTHING — use presence set (position-free)
    healthy_syms = set()
    for s in successes:
        healthy_syms.update(int(x) for x in enc[s.run_id])

    fseq = enc[failing.run_id]
    # nearest successful run by edit distance
    best_sid, best_cost = None, None
    for s in successes:
        cost, _, _ = needleman_wunsch(fseq, enc[s.run_id], alpha)
        if best_cost is None or cost < best_cost:
            best_cost, best_sid = cost, s.run_id

    sseq = enc[best_sid]
    _, fa, sa = needleman_wunsch(fseq, sseq, alpha)
    fp = 0
    for f, s in zip(fa.tolist(), sa.tolist()):
        if f != GAP and s != GAP and f != s and f not in healthy_syms:
            return Prediction(failing.run_id, failing.agent_at(fp), fp, 0.5,
                              meta={"baseline": "edit_nn", "sibling": best_sid})
        if f != GAP:
            fp += 1
    last = len(failing) - 1
    return Prediction(failing.run_id, failing.agent_at(last), last, 0.0,
                      meta={"baseline": "edit_nn", "fallback": "no_divergence"})


def make_llm_baseline(call_llm: Callable[[str], str], style: str = "all_at_once"
                      ) -> Attributor:
    """Adapter turning a text-in/text-out ``call_llm`` into a harness attributor.

    Not executed offline; provided so the Who&When LLM baselines (All-at-Once,
    Step-by-Step, Binary-Search, one-shot judge) plug into the same evaluation.
    """
    def _attr(pop: RunPopulation, failing: Trace,
              rng: np.random.RandomState) -> Prediction:
        transcript = "\n".join(f"[{s.index}] {s.agent}: {s.content or s.action}"
                               for s in failing.steps)
        prompt = (f"You are debugging a failed multi-agent run.\n{transcript}\n\n"
                  "Identify the single agent and step index responsible. "
                  "Answer as 'AGENT | STEP'.")
        raw = call_llm(prompt)
        agent, _, step = raw.partition("|")
        try:
            step_i = max(0, min(int(step.strip()), len(failing) - 1))
        except ValueError:
            step_i = len(failing) - 1
        return Prediction(failing.run_id, agent.strip() or failing.agent_at(0),
                          step_i, 0.5, meta={"baseline": f"llm_{style}"})
    return _attr


BASELINES: Dict[str, Attributor] = {
    "random": random_baseline,
    "first_step": first_step_baseline,
    "last_step": last_step_baseline,
    "most_active_agent": most_active_agent_baseline,
    "longest_agent": longest_agent_baseline,
    "edit_nn": edit_nn_baseline,
}


def run_baseline(name: str, pops: List[RunPopulation], seed: int = 0
                 ) -> Dict[str, Prediction]:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; have {sorted(BASELINES)}")
    fn = BASELINES[name]
    rng = np.random.RandomState(seed)
    preds: Dict[str, Prediction] = {}
    for pop in pops:
        for failing in pop.failures:
            preds[failing.run_id] = fn(pop, failing, rng)
    return preds
