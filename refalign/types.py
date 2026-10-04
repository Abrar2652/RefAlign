"""Core data model for multi-agent traces and failure-attribution labels.

Aligned with the Who&When schema (Zhang et al., ICML 2025):
  * a *trajectory* is a turn-based sequence in which exactly one agent acts per
    timestep: tau = (s0, a0, s1, a1, ..., sT);
  * a *decisive error* is the (agent, step) pair such that counterfactually
    correcting that action flips the run from failure to success.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Step:
    """One turn of a multi-agent run: a single agent takes one action."""

    index: int                      # 0-based position within the run (the "when")
    agent: str                      # acting agent (the "who")
    action: str                     # short canonical action label, e.g. "search", "code"
    content: str = ""               # full message / tool output (optional, for encoders/LLM judges)
    token: Optional[str] = None     # alignment symbol assigned by the encoder (filled later)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"Step.index must be >= 0, got {self.index}")


@dataclass
class Trace:
    """A full execution trace of one multi-agent run for one query."""

    run_id: str
    query_id: str
    steps: List[Step]
    success: bool                       # outcome-level label (did the run solve the query?)
    query: str = ""
    final_answer: str = ""
    agents: List[str] = field(default_factory=list)   # roster of agents in the system
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # steps must be contiguous and ordered; this is relied on everywhere downstream.
        for expected, s in enumerate(self.steps):
            if s.index != expected:
                raise ValueError(
                    f"Trace {self.run_id}: steps must be 0..N contiguous; "
                    f"position {expected} has index {s.index}"
                )
        if not self.agents:
            # derive roster from the steps if not supplied
            seen: List[str] = []
            for s in self.steps:
                if s.agent not in seen:
                    seen.append(s.agent)
            self.agents = seen

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def sequence(self) -> str:
        """The encoded token string (requires the encoder to have run)."""
        if any(s.token is None for s in self.steps):
            raise ValueError(f"Trace {self.run_id} not encoded; run an encoder first.")
        return "".join(s.token for s in self.steps)  # type: ignore[misc]

    def agent_at(self, step: int) -> str:
        return self.steps[step].agent


@dataclass
class Annotation:
    """Ground-truth failure attribution label for a single failing run."""

    query_id: str
    run_id: str
    responsible_agent: str          # the "who"
    decisive_step: int              # the "when" (0-based index into the run)
    reason: str = ""
    meta: Dict = field(default_factory=dict)   # e.g. {"fault_type": "skip"}


@dataclass
class Prediction:
    """A predicted failure attribution for a single failing run."""

    run_id: str
    agent: str
    step: int
    confidence: float = 1.0
    meta: Dict = field(default_factory=dict)


@dataclass
class RunPopulation:
    """All runs collected for one query — the unit the phylogeny is built over.

    A population mixes successful and failing runs. RefAlign's premise is that
    the successful runs define the "healthy lineage" against which a failing run's
    divergence is measured.
    """

    query_id: str
    runs: List[Trace]
    query: str = ""

    @property
    def successes(self) -> List[Trace]:
        return [r for r in self.runs if r.success]

    @property
    def failures(self) -> List[Trace]:
        return [r for r in self.runs if not r.success]

    def by_id(self, run_id: str) -> Trace:
        for r in self.runs:
            if r.run_id == run_id:
                return r
        raise KeyError(run_id)
