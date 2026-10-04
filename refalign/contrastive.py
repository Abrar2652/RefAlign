"""RefAlign — contrastive counterfactual failure attribution.

Runs of the same task form a *population of homologous sequences*. RefAlign
aligns a failing run to its nearest **successful sibling** (the counterfactual)
and localises the earliest aligned step where the failing run diverges from that
working run — comparing both the action *and the content value*. This is exactly
Who&When's decisive error (the earliest action whose correction flips failure to
success), and it is cheap: it reads two runs, not the whole trace space, and needs
no LLM. An optional LLM adjudication step refines the agent/step among the top
divergences when the content is fuzzier than a numeric value.

Two public entry points, both returning a :class:`Prediction`:
  * :func:`contrastive_localize`  — action+content alignment, no LLM (deterministic)
  * :func:`contrastive_llm`       — adds an LLM adjudication over the candidates
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import Trace, Prediction, RunPopulation
from .encode import Alphabet, GAP
from .msa import needleman_wunsch

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class ContrastiveConfig:
    sibling: str = "content"       # 'content' (action + value) | 'edit' (action only)
    value_tol: float = 1e-6        # numeric equality tolerance for content compare
    numeric_actions: Tuple[str, ...] = ("calc", "answer", "run", "code")
    top_k: int = 4                 # candidates handed to the LLM adjudicator


def _value(step) -> Optional[float]:
    nums = _NUM.findall((step.content or "").replace(",", ""))
    return float(nums[-1]) if nums else None


def _encode_actions(pop: RunPopulation) -> Tuple[Alphabet, Dict[str, np.ndarray]]:
    alpha = Alphabet(mode="action")
    seqs = {r.run_id: np.asarray([alpha.encode(s.agent, s.action) for s in r.steps],
                                 dtype=np.int64) for r in pop.runs}
    return alpha, seqs


def _sibling_distance(failing: Trace, s: Trace, alpha: Alphabet,
                      seqs: Dict[str, np.ndarray], cfg: ContrastiveConfig) -> float:
    """Distance from ``failing`` to a candidate sibling. 'edit' uses only the
    action sequence; 'content' additionally penalises value mismatches at aligned
    numeric steps, so that when successful runs are *diverse* (e.g. different valid
    orders), the sibling sharing the failing run's strategy is preferred."""
    cost, af, asb = needleman_wunsch(seqs[failing.run_id], seqs[s.run_id], alpha)
    if cfg.sibling != "content":
        return cost
    fp = sp = 0
    for x, y in zip(af.tolist(), asb.tolist()):
        if x != GAP and y != GAP:
            fstep, sstep = failing.steps[fp], s.steps[sp]
            if fstep.action in cfg.numeric_actions:
                vf, vs = _value(fstep), _value(sstep)
                if vf is not None and vs is not None:
                    # count matches vs mismatches (order-of-magnitude-agnostic)
                    cost += 0.0 if abs(vf - vs) <= cfg.value_tol else 1.0
            fp += 1
            sp += 1
        elif x != GAP:
            fp += 1
        else:
            sp += 1
    return cost


def _nearest_sibling(failing: Trace, successes: List[Trace], alpha: Alphabet,
                     seqs: Dict[str, np.ndarray],
                     cfg: ContrastiveConfig = ContrastiveConfig()) -> Trace:
    best, best_cost = successes[0], None
    for s in successes:
        d = _sibling_distance(failing, s, alpha, seqs, cfg)
        if best_cost is None or d < best_cost:
            best_cost, best = d, s
    return best


def _required_actions(pop: RunPopulation, thresh: float = 0.9) -> set:
    """Backbone actions: those whose per-run *count* is consistent across the
    healthy runs (the invariant skeleton). Optional steps such as notes appear a
    variable number of times and are excluded, so the alignment reconciles only
    the stable backbone — the key to robustness against non-identical siblings."""
    from collections import Counter
    succ = pop.successes
    if not succ:
        return set()
    all_actions = {s.action for r in succ for s in r.steps}
    required = set()
    for a in all_actions:
        counts = [sum(1 for s in r.steps if s.action == a) for r in succ]
        mode_count, mode_freq = Counter(counts).most_common(1)[0]
        if mode_count > 0 and mode_freq >= thresh * len(succ):
            required.add(a)
    return required


def _backbone(trace: Trace, val_alpha: Alphabet, required: set,
              numeric: Tuple[str, ...]):
    """Required-action subsequence of a run: (orig_indices, alignment_ids, steps).

    Alignment ids are *value-aware*: a numeric step is keyed by (action, value)
    so that matching prefixes lock in and a gap/divergence lands at the true
    fault (essential for omission faults, where all actions are identical). The
    ``val_alpha`` is shared across the two runs being aligned. Optional/benign
    steps are dropped so the alignment reconciles only the required backbone."""
    idx, ids, steps = [], [], []
    for s in trace.steps:
        if s.action not in required:
            continue
        idx.append(s.index)
        steps.append(s)
        if s.action in numeric:
            v = _value(s)
            ids.append(val_alpha.encode(s.action, f"{s.action}#{v}"))
        else:
            ids.append(val_alpha.encode(s.action, s.action))
    return idx, np.asarray(ids, dtype=np.int64), steps


def _candidates(failing: Trace, sibling: Trace, alpha: Alphabet,
                seqs: Dict[str, np.ndarray], cfg: ContrastiveConfig,
                required: set) -> List[Tuple[int, str, str]]:
    """Substantive divergences between the required backbones of the failing run
    and its sibling: value divergences, action substitutions, and required-action
    indels. Returns [(failing_step, kind, note)] in trace order."""
    val_alpha = Alphabet(mode="action")   # shared, value-aware alignment alphabet
    fidx, fids, fsteps = _backbone(failing, val_alpha, required, cfg.numeric_actions)
    sidx, sids, ssteps = _backbone(sibling, val_alpha, required, cfg.numeric_actions)
    if len(fids) == 0 or len(sids) == 0:
        return []
    _, af, asb = needleman_wunsch(fids, sids, val_alpha)
    out: List[Tuple[int, str, str, Optional[str]]] = []
    fp = sp = 0
    for x, y in zip(af.tolist(), asb.tolist()):
        if x != GAP and y != GAP:
            fstep, sstep = fsteps[fp], ssteps[sp]
            if fstep.action != sstep.action:
                out.append((fidx[fp], "action",
                            f"did '{fstep.action}' where the working run did '{sstep.action}'", None))
            elif fstep.action in cfg.numeric_actions:
                vf, vs = _value(fstep), _value(sstep)
                if vf is not None and vs is not None and abs(vf - vs) > cfg.value_tol:
                    out.append((fidx[fp], "value",
                                f"{fstep.action} gave {vf} where the working run got {vs}", None))
            fp += 1
            sp += 1
        elif x != GAP:      # failing has an extra required step (e.g. a repeat)
            out.append((fidx[fp], "insertion", f"extra '{fsteps[fp].action}' step", None))
            fp += 1
        else:               # failing skipped a required step; the culprit is who
            here = fidx[fp] if fp < len(fidx) else (len(failing) - 1)   # should have acted
            out.append((here, "deletion", f"skipped the required '{ssteps[sp].action}' step",
                        ssteps[sp].agent))
            sp += 1
    return out


def _fallback(failing: Trace) -> Prediction:
    last = len(failing) - 1
    return Prediction(failing.run_id, failing.agent_at(last), last, 0.0,
                      meta={"fallback": "no_sibling"})


def contrastive_localize(pop: RunPopulation, failing: Trace,
                         cfg: ContrastiveConfig = ContrastiveConfig()) -> Prediction:
    """Earliest content/structural divergence from the nearest successful sibling."""
    successes = pop.successes
    if not successes:
        return _fallback(failing)
    alpha, seqs = _encode_actions(pop)
    sib = _nearest_sibling(failing, successes, alpha, seqs, cfg)
    cands = _candidates(failing, sib, alpha, seqs, cfg, _required_actions(pop))
    if not cands:
        return _fallback(failing)
    step, kind, note, resp = cands[0]     # earliest divergence = decisive error
    step = min(step, len(failing) - 1)
    agent = resp or failing.agent_at(step)   # for a skip, who should have acted
    return Prediction(failing.run_id, agent, step, 1.0,
                      meta={"method": "contrastive", "kind": kind,
                            "sibling": sib.run_id, "note": note})


# ---------------------------------------------------------------------------
_ADJ_SYS = ("You debug multi-agent runs by comparison. You are given a FAILED run "
            "and a SUCCESSFUL run of the same task, plus a short list of the steps "
            "where they first diverge. Pick the single divergence that is the "
            "decisive error (the earliest one whose correction would fix the run). "
            "Answer exactly: STEP: <failing step index> | AGENT: <name>.")


def contrastive_llm(pop: RunPopulation, failing: Trace, llm,
                    cfg: ContrastiveConfig = ContrastiveConfig()) -> Prediction:
    """Contrastive localisation with an LLM adjudicating among the top-k candidates."""
    successes = pop.successes
    if not successes:
        return _fallback(failing)
    alpha, seqs = _encode_actions(pop)
    sib = _nearest_sibling(failing, successes, alpha, seqs, cfg)
    cands = _candidates(failing, sib, alpha, seqs, cfg, _required_actions(pop))
    if not cands:
        return _fallback(failing)
    top = cands[:cfg.top_k]

    def show(tr: Trace, lo: int, hi: int) -> str:
        return "\n".join(f"  [{s.index}] {s.agent}: {(s.content or s.action)[:120]}"
                         for s in tr.steps[lo:hi])

    lines = []
    for fstep, kind, note, _resp in top:
        lines.append(f"- failing step [{fstep}] ({kind}): {note}")
    user = (f"Task: {failing.query}\n\nSUCCESSFUL run:\n{show(sib, 0, len(sib))}\n\n"
            f"FAILED run:\n{show(failing, 0, len(failing))}\n\n"
            f"Divergences (earliest first):\n" + "\n".join(lines) +
            "\n\nWhich divergence is the decisive error?")
    raw = llm.chat(_ADJ_SYS, user)
    step = top[0][0]
    m = re.search(r"STEP\s*[:=]\s*(\d+)", raw, re.I)
    if m:
        cand_steps = [c[0] for c in top]
        s = int(m.group(1))
        step = s if s in cand_steps else min(cand_steps, key=lambda c: abs(c - s))
    step = min(step, len(failing) - 1)
    agent = failing.agent_at(step)
    m = re.search(r"AGENT\s*[:=]\s*([^\n|]+)", raw, re.I)
    if m:
        cand = m.group(1).strip()
        roster = [s.agent for s in failing.steps]
        if cand in roster:
            agent = cand
    return Prediction(failing.run_id, agent, step, 1.0,
                      meta={"method": "contrastive_llm", "sibling": sib.run_id})


def attribute_corpus_contrastive(pops: List[RunPopulation],
                                 cfg: ContrastiveConfig = ContrastiveConfig(),
                                 llm=None) -> Dict[str, Prediction]:
    fn = (lambda p, f: contrastive_llm(p, f, llm, cfg)) if llm is not None \
        else (lambda p, f: contrastive_localize(p, f, cfg))
    out: Dict[str, Prediction] = {}
    for pop in pops:
        for failing in pop.failures:
            out[failing.run_id] = fn(pop, failing)
    return out
