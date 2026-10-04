"""Evaluation harness: Who&When metrics with the statistical rigor A* readers
expect (research grounding, Section 4).

Metrics: Agent-Level Accuracy, Step-Level Accuracy, Step-Level Accuracy with
tolerance +-k. Reported with bootstrap 95% CIs and, for method-vs-baseline
comparisons, a paired bootstrap p-value and an exact McNemar test (the correct
paired test for two attributors scored on the *same* runs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .types import Prediction, Annotation


@dataclass
class Scored:
    """Per-run correctness vectors, aligned by run_id, for a single attributor."""

    run_ids: List[str]
    agent_correct: np.ndarray
    step_correct: np.ndarray
    step_tol: Dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.run_ids)

    def summary(self) -> Dict[str, float]:
        out = {
            "n": self.n,
            "agent_acc": float(self.agent_correct.mean()) if self.n else 0.0,
            "step_acc": float(self.step_correct.mean()) if self.n else 0.0,
        }
        for k, v in self.step_tol.items():
            out[f"step_acc_tol{k}"] = float(v.mean()) if self.n else 0.0
        return out


def score(preds: Dict[str, Prediction], anns: Dict[str, Annotation],
          tolerances: Sequence[int] = (1, 2, 3)) -> Scored:
    """Score predictions against ground truth on their common (annotated) runs."""
    run_ids = sorted(set(preds) & set(anns))
    ag = np.zeros(len(run_ids))
    st = np.zeros(len(run_ids))
    tol = {k: np.zeros(len(run_ids)) for k in tolerances}
    for i, rid in enumerate(run_ids):
        p, g = preds[rid], anns[rid]
        ag[i] = float(p.agent == g.responsible_agent)
        st[i] = float(p.step == g.decisive_step)
        for k in tolerances:
            tol[k][i] = float(abs(p.step - g.decisive_step) <= k)
    return Scored(run_ids, ag, st, tol)


# ---------------------------------------------------------------------------
def bootstrap_ci(x: np.ndarray, B: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap CI for the mean of ``x``."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, x.size, size=(B, x.size))
    means = x[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(x.mean()), lo, hi


def paired_bootstrap(x: np.ndarray, y: np.ndarray, B: int = 5000,
                     seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap on the per-run difference x - y (shared items).

    Returns the observed difference, a 95% CI, and a two-sided p-value for
    H0: mean(x) == mean(y).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    assert x.shape == y.shape, "paired test needs aligned run vectors"
    d = x - y
    n = d.size
    if n == 0:
        return {"diff": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p_value": 1.0}
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(B, n))
    boot = d[idx].mean(axis=1)
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    # two-sided p: proportion of bootstrap means on the opposite side of 0
    p = 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    return {"diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
            "p_value": float(min(1.0, p))}


def mcnemar(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Exact McNemar test on paired binary correctness (x vs y)."""
    x = np.asarray(x, int)
    y = np.asarray(y, int)
    b = int(np.sum((x == 1) & (y == 0)))   # x right, y wrong
    c = int(np.sum((x == 0) & (y == 1)))   # x wrong, y right
    nd = b + c
    if nd == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    # exact two-sided binomial p on discordant pairs (p=0.5)
    from math import comb
    k = min(b, c)
    tail = sum(comb(nd, i) for i in range(0, k + 1)) / (2 ** nd)
    p = min(1.0, 2.0 * tail)
    return {"b": b, "c": c, "p_value": float(p)}


# ---------------------------------------------------------------------------
def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size between two proportions (|h|: .2 small, .5 med, .8 large)."""
    import math
    phi = lambda p: 2.0 * math.asin(math.sqrt(min(1.0, max(0.0, p))))
    return float(phi(p1) - phi(p2))


def holm_bonferroni(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, dict]:
    """Holm-Bonferroni step-down correction over a family of p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, dict] = {}
    prev = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))   # enforce monotonicity
        prev = adj
        out[name] = {"p_raw": float(p), "p_holm": float(adj),
                     "reject": bool(adj < alpha)}
    return out


@dataclass
class Row:
    name: str
    summary: Dict[str, float]
    ci: Dict[str, Tuple[float, float]]


def leaderboard(named_preds: Dict[str, Dict[str, Prediction]],
                anns: Dict[str, Annotation], tolerances: Sequence[int] = (1,),
                B: int = 2000, seed: int = 0) -> List[Row]:
    """Score several attributors and attach bootstrap CIs for each metric."""
    rows: List[Row] = []
    for name, preds in named_preds.items():
        sc = score(preds, anns, tolerances)
        summ = sc.summary()
        ci: Dict[str, Tuple[float, float]] = {}
        _, lo, hi = bootstrap_ci(sc.agent_correct, B=B, seed=seed)
        ci["agent_acc"] = (lo, hi)
        _, lo, hi = bootstrap_ci(sc.step_correct, B=B, seed=seed)
        ci["step_acc"] = (lo, hi)
        for k in tolerances:
            _, lo, hi = bootstrap_ci(sc.step_tol[k], B=B, seed=seed)
            ci[f"step_acc_tol{k}"] = (lo, hi)
        rows.append(Row(name, summ, ci))
    return rows


def format_leaderboard(rows: List[Row], tolerances: Sequence[int] = (1,)) -> str:
    """Render a leaderboard as a fixed-width table with 95% CIs."""
    cols = ["agent_acc", "step_acc"] + [f"step_acc_tol{k}" for k in tolerances]
    head = f"{'method':22s} {'n':>4s} " + " ".join(f"{c:>22s}" for c in cols)
    lines = [head, "-" * len(head)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.summary.get(c, 0.0)
            lo, hi = r.ci.get(c, (0.0, 0.0))
            cells.append(f"{v:5.3f} [{lo:.3f},{hi:.3f}]")
        lines.append(f"{r.name:22s} {int(r.summary['n']):4d} " +
                     " ".join(f"{x:>22s}" for x in cells))
    return "\n".join(lines)


def compare(method: Scored, baseline: Scored, metric: str = "agent") -> Dict[str, float]:
    """Paired comparison (bootstrap + McNemar) of two attributors on shared runs."""
    common = sorted(set(method.run_ids) & set(baseline.run_ids))
    mi = {r: i for i, r in enumerate(method.run_ids)}
    bi = {r: i for i, r in enumerate(baseline.run_ids)}
    pick = {"agent": ("agent_correct",), "step": ("step_correct",)}[metric]
    mvec = getattr(method, pick[0])[[mi[r] for r in common]]
    bvec = getattr(baseline, pick[0])[[bi[r] for r in common]]
    out = paired_bootstrap(mvec, bvec)
    out.update({f"mcnemar_{k}": v for k, v in mcnemar(mvec, bvec).items()})
    out["cohens_h"] = cohens_h(float(mvec.mean()), float(bvec.mean()))
    out["n_common"] = len(common)
    return out
