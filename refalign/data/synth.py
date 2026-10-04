"""Synthetic fault-injected multi-agent corpus with ground-truth attribution.

Follows the AgentTrace / AEGIS / MAS-FIRE controlled-injection protocol
(research grounding, Section 6): build a deterministically successful backbone,
inject a fault at a *known* (agent, step), and let it propagate downstream.

Design that makes the phylogenetics pipeline *necessary* (not trivially solvable
by a lookup):

  * a run is a **required conserved backbone** (each phase a fixed canonical
    (agent, action)) interleaved with sparse **optional** steps that healthy runs
    include only sometimes. The optional steps create indel/length diversity, so
    the failing run's steps must be *aligned* to the backbone before a column can
    be compared — a global symbol lookup does not suffice;
  * faults are **in-repertoire but position-wrong**: the offending agent emits an
    action it legitimately uses at *another* phase (e.g. a Coder runs before it
    codes), or skips / repeats a conserved step. Because the offending symbol
    still occurs elsewhere in healthy runs, only *position-aware* consensus (via
    the MSA + phylogeny) can localise it.

Two regimes mirror Who&When: "algo" (~7-10 steps) and "hand" (~40-55 steps, with
repeated cycles that stress alignment).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..types import Step, Trace, Annotation, RunPopulation

AGENTS: List[str] = ["Planner", "Researcher", "Coder", "Verifier"]

# Canonical (required) actions each agent may take across the template. All of
# these are legitimate healthy symbols; a fault reuses one at the wrong phase.
GOOD_ACTIONS: Dict[str, List[str]] = {
    "Planner": ["plan", "decompose"],
    "Researcher": ["search", "read", "extract"],
    "Coder": ["code", "run", "debug"],
    "Verifier": ["check", "test", "finalize"],
}

# Optional (benign, off-backbone) actions — create alignment diversity only.
OPTIONAL_ACTIONS: Dict[str, str] = {
    "Planner": "reflect",
    "Researcher": "note",
    "Coder": "lint",
    "Verifier": "review",
}

# Retained for backwards-compatible imports / documentation of fault vocabulary.
BAD_ACTIONS: Dict[str, List[str]] = {a: [] for a in AGENTS}


def _algo_backbone() -> List[Tuple[str, str]]:
    return [
        ("Planner", "plan"),
        ("Researcher", "search"),
        ("Researcher", "read"),
        ("Coder", "code"),
        ("Coder", "run"),
        ("Verifier", "check"),
        ("Verifier", "finalize"),
    ]


def _hand_backbone() -> List[Tuple[str, str]]:
    tmpl: List[Tuple[str, str]] = [("Planner", "plan"), ("Planner", "decompose")]
    for _ in range(6):
        tmpl += [
            ("Researcher", "search"),
            ("Researcher", "read"),
            ("Researcher", "extract"),
            ("Coder", "code"),
            ("Coder", "run"),
            ("Coder", "debug"),
            ("Verifier", "check"),
        ]
    tmpl += [("Verifier", "test"), ("Verifier", "finalize")]
    return tmpl


@dataclass
class SyntheticConfig:
    n_queries: int = 20
    runs_per_query: int = 8
    failure_fraction: float = 0.5
    regime: str = "algo"              # "algo" | "hand"
    p_optional: float = 0.35          # prob of inserting each optional step (diversity)
    propagate: float = 0.6            # prob a fault corrupts each later required step
    seed: int = 0
    extra_agents: List[str] = field(default_factory=list)


def _backbone(regime: str) -> List[Tuple[str, str]]:
    if regime == "algo":
        return _algo_backbone()
    if regime == "hand":
        return _hand_backbone()
    raise ValueError(f"unknown regime {regime!r} (use 'algo' or 'hand')")


def _emit_healthy(backbone: List[Tuple[str, str]], rng: np.random.RandomState,
                  cfg: SyntheticConfig) -> Tuple[List[Tuple[str, str]], List[int]]:
    """Emit a healthy run: canonical backbone + optional benign steps.

    Returns (run, required_positions) where required_positions[i] is the emitted
    index of the i-th backbone step.
    """
    run: List[Tuple[str, str]] = []
    req_pos: List[int] = []
    for agent, action in backbone:
        # optional benign step before this backbone step
        if rng.rand() < cfg.p_optional:
            run.append((agent, OPTIONAL_ACTIONS[agent]))
        req_pos.append(len(run))
        run.append((agent, action))
    return run, req_pos


def _agent_alt_action(agent: str, backbone: List[Tuple[str, str]],
                      current: str) -> Optional[str]:
    """Another canonical action this agent uses elsewhere in the backbone."""
    alts = {act for ag, act in backbone if ag == agent and act != current}
    return sorted(alts)[0] if alts else None


def _inject_fault(run: List[Tuple[str, str]], req_pos: List[int],
                  backbone: List[Tuple[str, str]], rng: np.random.RandomState,
                  cfg: SyntheticConfig) -> Tuple[List[Tuple[str, str]], int, str, str]:
    """Inject an in-repertoire, position-wrong fault at a required step.

    Returns (corrupted_run, decisive_step_index, responsible_agent, fault_kind).
    """
    # choose an interior required step (not the first plan, not the final step)
    candidates = list(range(1, len(backbone) - 1))
    rng.shuffle(candidates)

    for bi in candidates:
        agent, action = backbone[bi]
        alt = _agent_alt_action(agent, backbone, action)
        kind = rng.choice(["substitute", "skip", "repeat"])
        pos = req_pos[bi]
        corrupted = list(run)

        if kind == "repeat":
            # step repetition: the previous agent re-acts in place of this step,
            # so the required action never happens (in-repertoire, position-wrong).
            prev_agent, prev_act = run[pos - 1]
            corrupted[pos] = (prev_agent, prev_act)
            decisive = pos
            resp = prev_agent
        elif kind == "substitute" and alt is not None:
            corrupted[pos] = (agent, alt)               # wrong-context action
            decisive = pos
            resp = agent
        else:  # skip: delete the conserved step (omission)
            del corrupted[pos]
            decisive = min(pos, len(corrupted) - 1)         # gap manifests here
            resp = agent                                     # who should have acted

        # propagate: corrupt later required steps (still in-repertoire)
        # recompute later required emitted positions after any indel
        shift = -1 if kind == "skip" else 0
        for bj in range(bi + 1, len(backbone)):
            if rng.rand() >= cfg.propagate:
                continue
            p = req_pos[bj] + shift
            if not (0 <= p < len(corrupted)):
                continue
            ag2, act2 = backbone[bj]
            alt2 = _agent_alt_action(ag2, backbone, act2)
            if alt2 is not None:
                corrupted[p] = (ag2, alt2)
        return corrupted, decisive, resp, str(kind)

    # no injectable step found (degenerate): corrupt the last interior step
    pos = req_pos[len(backbone) // 2]
    agent = run[pos][0]
    corrupted = list(run)
    corrupted[pos] = (agent, OPTIONAL_ACTIONS[agent])
    return corrupted, pos, agent, "substitute"


def _to_trace(run_id: str, query_id: str, query: str,
              tuples: List[Tuple[str, str]], success: bool) -> Trace:
    steps = [Step(index=i, agent=a, action=act, content=f"{a}:{act}")
             for i, (a, act) in enumerate(tuples)]
    return Trace(run_id=run_id, query_id=query_id, steps=steps, success=success,
                 query=query, agents=list(AGENTS))


def generate_corpus(cfg: SyntheticConfig
                    ) -> Tuple[List[RunPopulation], Dict[str, Annotation]]:
    """Generate a synthetic corpus.

    Returns (populations, annotations) where ``annotations`` maps a failing
    run_id to its ground-truth :class:`Annotation`.
    """
    rng = np.random.RandomState(cfg.seed)
    backbone = _backbone(cfg.regime)

    populations: List[RunPopulation] = []
    annotations: Dict[str, Annotation] = {}

    for q in range(cfg.n_queries):
        qid = f"q{q:03d}"
        query = f"synthetic task {q} ({cfg.regime})"
        n_fail = int(round(cfg.runs_per_query * cfg.failure_fraction))
        n_fail = min(n_fail, cfg.runs_per_query - 1)  # keep >=1 success
        runs: List[Trace] = []

        for r in range(cfg.runs_per_query):
            rid = f"{qid}_r{r:02d}"
            run, req_pos = _emit_healthy(backbone, rng, cfg)
            if r < n_fail:
                corrupted, f, agent, kind = _inject_fault(run, req_pos, backbone, rng, cfg)
                runs.append(_to_trace(rid, qid, query, corrupted, success=False))
                annotations[rid] = Annotation(
                    query_id=qid, run_id=rid, responsible_agent=agent,
                    decisive_step=f, reason=f"injected {kind} fault at step {f} by {agent}",
                    meta={"fault_type": kind},
                )
            else:
                runs.append(_to_trace(rid, qid, query, run, success=True))

        order = rng.permutation(len(runs))
        runs = [runs[i] for i in order]
        populations.append(RunPopulation(query_id=qid, runs=runs, query=query))

    return populations, annotations
