import numpy as np

from refalign import evaluate as ev
from refalign.evostat import (blombergs_K, pagels_lambda, continuous_signal_test,
                                knn_purity)
from refalign.trees import upgma
from refalign.geometry import patristic_distance_matrix


def _clustered_patristic():
    # two tight clades {A,B} and {C,D} far apart -> strong structure
    labels = ["A", "B", "C", "D"]
    D = np.array([[0, 1, 9, 9],
                  [1, 0, 9, 9],
                  [9, 9, 0, 1],
                  [9, 9, 1, 0]], float)
    tree = upgma(labels, D)
    leaves, P = patristic_distance_matrix(tree)
    return leaves, P


def test_cohens_h_signs_and_zero():
    assert ev.cohens_h(0.5, 0.5) == 0.0
    assert ev.cohens_h(0.8, 0.2) > 0
    assert ev.cohens_h(0.2, 0.8) < 0


def test_holm_bonferroni_monotone_and_reject():
    pv = {"a": 0.001, "b": 0.04, "c": 0.20, "d": 0.9}
    out = ev.holm_bonferroni(pv, alpha=0.05)
    # adjusted p-values are non-decreasing in raw order
    order = sorted(out.items(), key=lambda kv: kv[1]["p_raw"])
    adj = [o[1]["p_holm"] for o in order]
    assert adj == sorted(adj)
    assert out["a"]["reject"] is True         # 0.001*4 = 0.004 < 0.05
    assert out["d"]["reject"] is False


def test_blomberg_K_signal_vs_none():
    leaves, P = _clustered_patristic()
    # trait aligned with clades -> strong signal (K larger); random -> weaker
    aligned = np.array([1.0 if l in ("A", "B") else 0.0 for l in leaves])
    k_signal = blombergs_K(P, aligned)
    antisig = np.array([1.0, 0.0, 1.0, 0.0])  # crosses the clades
    k_anti = blombergs_K(P, antisig)
    assert np.isfinite(k_signal) and np.isfinite(k_anti)
    assert k_signal > k_anti


def test_pagels_lambda_range():
    leaves, P = _clustered_patristic()
    aligned = np.array([1.0 if l in ("A", "B") else 0.0 for l in leaves])
    lam = pagels_lambda(P, aligned)
    assert 0.0 <= lam <= 1.0


def test_continuous_signal_pvalue_bounds():
    leaves, P = _clustered_patristic()
    x = np.array([1.0 if l in ("A", "B") else 0.0 for l in leaves])
    out = continuous_signal_test(P, x, "K", n_perm=199, seed=0)
    assert 0.0 <= out["p_value"] <= 1.0


def test_knn_purity_perfect_and_random():
    # block-diagonal distance: same-label points are nearest -> purity 1.0
    D = np.array([[0, 1, 9, 9],
                  [1, 0, 9, 9],
                  [9, 9, 0, 1],
                  [9, 9, 1, 0]], float)
    labels = ["x", "x", "y", "y"]
    assert knn_purity(D, labels, k=1) == 1.0
