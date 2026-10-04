"""Evolutionary / phylogenetic statistics on populations of traces.

  * :func:`column_conservation` — per-column conservation of the MSA (how
    reliably the healthy lineage reproduces each step);
  * :func:`bootstrap_support` — Felsenstein bootstrap: resample alignment columns,
    re-infer the tree, and report the fraction of replicates recovering each
    clade (the standard clade-support statistic);
  * :func:`failure_signal_permutation` — a permutation test for whether failing
    runs *cluster* on the phylogeny (a phylogenetic-signal analogue for the
    binary success/failure trait). Significant clustering is the interpretable,
    characterisation-level pay-off the research grounding attributes to the
    tree-space view even when per-run localisation does not beat edit distance.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional

import numpy as np

from .encode import Alphabet, GAP
from .msa import MultipleAlignment, center_star_msa
from .trees import build_tree, distance_from_msa
from .geometry import patristic_distance_matrix
from .types import RunPopulation


def column_conservation(msa: MultipleAlignment,
                        labels: Optional[List[str]] = None) -> np.ndarray:
    """Per-column conservation = frequency of the most common non-gap symbol."""
    labels = labels or msa.labels
    rows = [msa.labels.index(l) for l in labels]
    M = msa.matrix[rows]
    ncol = M.shape[1]
    out = np.zeros(ncol)
    for c in range(ncol):
        col = M[:, c]
        col = col[col != GAP]
        if col.size == 0:
            out[c] = 0.0
            continue
        _, counts = np.unique(col, return_counts=True)
        out[c] = counts.max() / len(rows)
    return out


def bootstrap_support(labels: List[str], seqs: List[np.ndarray], alphabet: Alphabet,
                      tree_method: str = "nj", B: int = 100, seed: int = 0
                      ) -> Dict[FrozenSet[str], float]:
    """Felsenstein bootstrap clade support over resampled alignment columns."""
    msa = center_star_msa(labels, seqs, alphabet)
    M = msa.matrix
    ncol = M.shape[1]
    ref_L, ref_D = distance_from_msa(msa)
    ref_tree = build_tree(ref_L, ref_D, method=tree_method)
    ref_splits = ref_tree.bipartitions()

    counts: Dict[FrozenSet[str], int] = {s: 0 for s in ref_splits}
    rng = np.random.RandomState(seed)
    for _ in range(B):
        cols = rng.randint(0, ncol, size=ncol)
        Mb = M[:, cols]
        boot = MultipleAlignment(list(msa.labels), Mb, alphabet)
        L, D = distance_from_msa(boot)
        tb = build_tree(L, D, method=tree_method)
        got = tb.bipartitions()
        for s in ref_splits:
            if s in got:
                counts[s] += 1
    return {s: counts[s] / B for s in ref_splits}


def failure_signal_permutation(pop: RunPopulation, alphabet: Alphabet,
                               seqs: Dict[str, np.ndarray], tree_method: str = "nj",
                               n_perm: int = 1000, seed: int = 0
                               ) -> Dict[str, float]:
    """Do failing runs cluster on the phylogeny more than chance?

    Statistic: mean patristic distance between same-outcome leaf pairs (lower =>
    stronger clustering / phylogenetic signal). Compared to a null from permuting
    the success/failure labels across leaves.
    """
    labels = [r.run_id for r in pop.runs]
    S = [seqs[l] for l in labels]
    msa = center_star_msa(labels, S, alphabet)
    L, D = distance_from_msa(msa)
    tree = build_tree(L, D, method=tree_method)
    leaves, P = patristic_distance_matrix(tree)

    trait = np.array([0 if pop.by_id(l).success else 1 for l in leaves])
    if trait.sum() < 2 or (len(trait) - trait.sum()) < 2:
        return {"observed": float("nan"), "p_value": float("nan"),
                "n_failure": int(trait.sum())}

    def stat(t: np.ndarray) -> float:
        tot = 0.0
        cnt = 0
        n = len(t)
        for i in range(n):
            for j in range(i + 1, n):
                if t[i] == t[j]:
                    tot += P[i, j]
                    cnt += 1
        return tot / cnt if cnt else 0.0

    obs = stat(trait)
    rng = np.random.RandomState(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = stat(rng.permutation(trait))
    # clustering => observed smaller than null
    p = float((null <= obs + 1e-12).mean())
    return {"observed": float(obs), "null_mean": float(null.mean()),
            "p_value": p, "n_failure": int(trait.sum())}


# ---------------------------------------------------------------------------
# Phylogenetic signal of a continuous trait (Blomberg's K, Pagel's lambda)
# ---------------------------------------------------------------------------
def _vcv_from_patristic(P: np.ndarray) -> np.ndarray:
    """Phylogenetic variance-covariance matrix from an (ultrametric) patristic
    distance matrix: C[i,j] = shared root-to-MRCA height = T - d(i,j)/2, with
    C[i,i] = T = tip height (0.5 * max patristic)."""
    T = 0.5 * float(P.max()) if P.size else 0.0
    C = T - 0.5 * P
    np.fill_diagonal(C, T)
    # tiny ridge for numerical PD-ness
    C = C + 1e-9 * np.eye(C.shape[0])
    return C


def blombergs_K(P: np.ndarray, x: np.ndarray) -> float:
    """Blomberg's K phylogenetic signal (K=1 => Brownian expectation on the tree,
    K<1 => less signal, K>1 => more), computed from the patristic VCV."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return float("nan")
    C = _vcv_from_patristic(P)
    Cinv = np.linalg.pinv(C)
    one = np.ones(n)
    denom = float(one @ Cinv @ one)
    a_hat = float(one @ Cinv @ x) / denom
    d = x - a_hat
    mse0 = float(d @ d) / (n - 1)
    mse = float(d @ Cinv @ d) / (n - 1)
    if mse <= 0:
        return float("nan")
    expected = (np.trace(C) - n / denom) / (n - 1)
    return float((mse0 / mse) / expected) if expected > 0 else float("nan")


def pagels_lambda(P: np.ndarray, x: np.ndarray, grid: int = 41) -> float:
    """Maximum-likelihood Pagel's lambda in [0,1] (0 = no signal, 1 = Brownian),
    via a grid search over the lambda-scaled VCV."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return float("nan")
    C = _vcv_from_patristic(P)
    diag = np.diag(np.diag(C))
    off = C - diag
    one = np.ones(n)

    def loglik(lam: float) -> float:
        Cl = diag + lam * off + 1e-9 * np.eye(n)
        try:
            Cinv = np.linalg.inv(Cl)
            sign, logdet = np.linalg.slogdet(Cl)
        except np.linalg.LinAlgError:
            return -np.inf
        if sign <= 0:
            return -np.inf
        denom = float(one @ Cinv @ one)
        a_hat = float(one @ Cinv @ x) / denom
        d = x - a_hat
        s2 = float(d @ Cinv @ d) / n
        if s2 <= 0:
            return -np.inf
        return -0.5 * (n * np.log(2 * np.pi * s2) + logdet + n)

    lams = np.linspace(0.0, 1.0, grid)
    best = max(lams, key=loglik)
    return float(best)


def continuous_signal_test(P: np.ndarray, x: np.ndarray, stat: str = "K",
                           n_perm: int = 999, seed: int = 0) -> Dict[str, float]:
    """Permutation test for phylogenetic signal of a continuous trait.

    ``stat`` in {"K","lambda"}. p = P(permuted statistic >= observed) (one-sided,
    testing for *more* signal than random tip assignment).
    """
    x = np.asarray(x, float)
    fn = blombergs_K if stat == "K" else pagels_lambda
    obs = fn(P, x)
    if not np.isfinite(obs):
        return {"observed": float("nan"), "p_value": float("nan"), "stat": stat}
    rng = np.random.RandomState(seed)
    ge = 1  # include observed
    total = 1
    for _ in range(n_perm):
        v = fn(P, rng.permutation(x))
        if np.isfinite(v):
            total += 1
            if v >= obs - 1e-12:
                ge += 1
    return {"observed": float(obs), "p_value": float(ge / total), "stat": stat}


def knn_purity(D: np.ndarray, labels: List, k: int = 3) -> float:
    """Mean fraction of each point's k nearest neighbours sharing its label.

    Used to compare how well a distance (tree-space patristic vs raw edit)
    organises failing runs by failure mode.
    """
    n = len(labels)
    if n <= k:
        k = max(1, n - 1)
    tot = 0.0
    for i in range(n):
        order = np.argsort(D[i])
        nbrs = [j for j in order if j != i][:k]
        if not nbrs:
            continue
        tot += np.mean([labels[j] == labels[i] for j in nbrs])
    return float(tot / n) if n else 0.0
