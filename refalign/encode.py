"""Trace -> sequence encoding: the (novelty-critical) traces-as-sequences map.

Each turn of a run is mapped to one discrete *symbol* drawn from an
"agent-action alphabet". A run then becomes a sequence of symbols, exactly the
object a multiple-sequence aligner consumes. Three granularities are supported:

    "agent_action"  symbol = (agent, action)   -- richest; default
    "action"        symbol = action            -- what happened, ignoring who
    "agent"         symbol = agent             -- who acted, ignoring what

Because failure attribution needs *both* who and when, the default keeps agent
and action together but also exposes a substitution cost that gives partial
credit when two symbols share an agent (same agent, different action) — a
soft substitution matrix in the spirit of BLOSUM/PAM.

Sequences are represented as ``numpy`` int arrays of symbol ids; the gap symbol
used by the aligner is the module constant :data:`GAP` (``-1``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import Trace, RunPopulation

GAP = -1  # gap symbol id used across MSA / distance code


@dataclass
class Alphabet:
    """Bidirectional symbol <-> id map that remembers (agent, action) structure."""

    mode: str = "agent_action"
    sym2id: Dict[str, int] = field(default_factory=dict)
    id2sym: List[str] = field(default_factory=list)
    id2agent: List[Optional[str]] = field(default_factory=list)
    id2action: List[Optional[str]] = field(default_factory=list)

    def _symbol_for(self, agent: str, action: str) -> Tuple[str, Optional[str], Optional[str]]:
        if self.mode == "agent_action":
            return f"{agent}→{action}", agent, action
        if self.mode == "action":
            return action, None, action
        if self.mode == "agent":
            return agent, agent, None
        raise ValueError(f"unknown encoding mode {self.mode!r}")

    def encode(self, agent: str, action: str) -> int:
        sym, ag, act = self._symbol_for(agent, action)
        if sym not in self.sym2id:
            i = len(self.id2sym)
            self.sym2id[sym] = i
            self.id2sym.append(sym)
            self.id2agent.append(ag)
            self.id2action.append(act)
        return self.sym2id[sym]

    def __len__(self) -> int:
        return len(self.id2sym)

    def symbol(self, i: int) -> str:
        return "-" if i == GAP else self.id2sym[i]

    def substitution_cost(self, a: int, b: int) -> float:
        """Cost of aligning symbol ``a`` against symbol ``b`` (0 == identical)."""
        if a == b:
            return 0.0
        if a == GAP or b == GAP:
            return 1.0
        # partial credit when the acting agent matches (only meaningful in
        # agent_action mode, where id2agent is populated).
        ag_a, ag_b = self.id2agent[a], self.id2agent[b]
        if ag_a is not None and ag_a == ag_b:
            return 0.5
        return 1.0


def encode_trace(trace: Trace, alphabet: Alphabet) -> np.ndarray:
    """Encode one trace to an int array, stamping each Step.token in place."""
    ids: List[int] = []
    for s in trace.steps:
        i = alphabet.encode(s.agent, s.action)
        s.token = alphabet.id2sym[i]
        ids.append(i)
    return np.asarray(ids, dtype=np.int64)


def encode_population(pop: RunPopulation, mode: str = "agent_action",
                      alphabet: Optional[Alphabet] = None
                      ) -> Tuple[Alphabet, Dict[str, np.ndarray]]:
    """Encode every run in a population, sharing one alphabet.

    Returns (alphabet, {run_id: int-sequence}).
    """
    if alphabet is None:
        alphabet = Alphabet(mode=mode)
    elif alphabet.mode != mode:
        raise ValueError(f"alphabet.mode={alphabet.mode!r} != requested {mode!r}")
    seqs: Dict[str, np.ndarray] = {}
    for run in pop.runs:
        seqs[run.run_id] = encode_trace(run, alphabet)
    return alphabet, seqs
