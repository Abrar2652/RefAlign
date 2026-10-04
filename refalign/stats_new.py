"""Statistics and file-schema helpers for the the paper follow-up experiments.

All p-values and intervals come from ``scipy.stats`` / ``statsmodels`` or from
the existing helpers in :mod:`refalign.evaluate`; nothing is hand-rolled.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats

from . import evaluate as ev

REQUIRED_PAIRED_KEYS = ("agent_pred", "step_pred", "agent_correct", "step_correct", "tokens")


def wilson(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (scipy)."""
    if n == 0:
        return (0.0, 1.0)
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=conf, method="wilson")
    return (float(ci.low), float(ci.high))


def acc_record(correct: Sequence[bool]) -> Dict[str, float]:
    c = int(np.sum(np.asarray(correct, bool)))
    n = int(len(correct))
    lo, hi = wilson(c, n)
    return {"c": c, "n": n, "acc": (c / n if n else 0.0), "wilson_lo": lo, "wilson_hi": hi}


def mcnemar_exact(x: Sequence[bool], y: Sequence[bool]) -> Dict[str, float]:
    """Exact (binomial) two-sided McNemar test on paired binary outcomes.

    ``b`` = x right & y wrong, ``c`` = x wrong & y right. The p-value is scipy's
    exact two-sided binomial test of ``min(b, c)`` out of ``b + c`` at p=0.5, and
    is cross-checked against :func:`refalign.evaluate.mcnemar`.
    """
    x = np.asarray(x, bool)
    y = np.asarray(y, bool)
    assert x.shape == y.shape, "paired test needs aligned outcome vectors"
    b = int(np.sum(x & ~y))
    c = int(np.sum(~x & y))
    if b + c == 0:
        p = 1.0
    else:
        p = float(stats.binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
    ref = ev.mcnemar(x.astype(int), y.astype(int))["p_value"]
    assert abs(ref - p) < 1e-9, (ref, p)
    return {"b": b, "c": c, "n_discordant": b + c, "p_value": p,
            "n": int(x.size), "acc_x": float(x.mean()) if x.size else 0.0,
            "acc_y": float(y.mean()) if y.size else 0.0,
            "cohens_h": ev.cohens_h(float(x.mean()) if x.size else 0.0,
                                    float(y.mean()) if y.size else 0.0)}


def holm(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, dict]:
    return ev.holm_bonferroni(pvalues, alpha)


def fisher_2x2(fail_a: int, n_a: int, fail_b: int, n_b: int) -> Dict[str, float]:
    """Two-sided Fisher exact test on failure counts of two independent arms."""
    table = [[fail_a, n_a - fail_a], [fail_b, n_b - fail_b]]
    odds, p = stats.fisher_exact(table, alternative="two-sided")
    return {"odds_ratio": float(odds) if np.isfinite(odds) else None, "p_value": float(p),
            "table": table}


def diff_ci(fail_a: int, n_a: int, fail_b: int, n_b: int, conf: float = 0.95) -> Dict[str, float]:
    """Difference in proportions (arm b minus arm a) with a Newcombe score CI."""
    from statsmodels.stats.proportion import confint_proportions_2indep
    lo, hi = confint_proportions_2indep(fail_b, n_b, fail_a, n_a, method="newcomb",
                                        compare="diff", alpha=1 - conf)
    return {"diff": (fail_b / n_b if n_b else 0.0) - (fail_a / n_a if n_a else 0.0),
            "ci_lo": float(lo), "ci_hi": float(hi)}


def validate_paired(obj: dict) -> int:
    """Validate the paired-outcome file schema
    ``{judge: {cell: {method: {run_id: {agent_pred, step_pred, agent_correct,
    step_correct, tokens, ...}}}}}``. Returns the number of run records checked
    and raises ``ValueError`` on the first violation."""
    n = 0
    if not isinstance(obj, dict) or not obj:
        raise ValueError("paired file must be a non-empty dict keyed by judge")
    for judge, cells in obj.items():
        if not isinstance(cells, dict):
            raise ValueError(f"{judge}: cells must be a dict")
        for cell, methods in cells.items():
            if not isinstance(methods, dict):
                raise ValueError(f"{judge}/{cell}: methods must be a dict")
            for method, runs in methods.items():
                if not isinstance(runs, dict) or not runs:
                    raise ValueError(f"{judge}/{cell}/{method}: runs must be a non-empty dict")
                for rid, rec in runs.items():
                    for k in REQUIRED_PAIRED_KEYS:
                        if k not in rec:
                            raise ValueError(f"{judge}/{cell}/{method}/{rid}: missing {k}")
                    if not isinstance(rec["step_pred"], int):
                        raise ValueError(f"{judge}/{cell}/{method}/{rid}: step_pred must be int")
                    if not isinstance(rec["agent_correct"], bool) or not isinstance(rec["step_correct"], bool):
                        raise ValueError(f"{judge}/{cell}/{method}/{rid}: *_correct must be bool")
                    if not isinstance(rec["tokens"], int) or rec["tokens"] < 0:
                        raise ValueError(f"{judge}/{cell}/{method}/{rid}: tokens must be int >= 0")
                    n += 1
    return n


def paired_vectors(runs_a: Dict[str, dict], runs_b: Dict[str, dict], key: str = "step_correct"
                   ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rids = sorted(set(runs_a) & set(runs_b))
    a = np.asarray([runs_a[r][key] for r in rids], bool)
    b = np.asarray([runs_b[r][key] for r in rids], bool)
    return a, b, rids


def fmt_acc(rec: Dict[str, float], digits: int = 2) -> str:
    return f"{rec['acc']:.{digits}f} [{rec['wilson_lo']:.{digits}f}, {rec['wilson_hi']:.{digits}f}]"


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"
