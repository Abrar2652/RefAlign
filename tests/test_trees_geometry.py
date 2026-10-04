import math
import numpy as np
import pytest

from refalign.trees import neighbor_joining, upgma, distance_from_msa
from refalign.geometry import (TreePoint, bhv_distance, geodesic_interpolate,
                                 frechet_mean, frechet_variance, rf_distance,
                                 patristic_distance_matrix, splits_compatible)


def _textbook_D():
    labels = ["A", "B", "C", "D", "E"]
    D = np.array([
        [0, 5, 9, 9, 8],
        [5, 0, 10, 10, 9],
        [9, 10, 0, 8, 7],
        [9, 10, 8, 0, 3],
        [8, 9, 7, 3, 0]], float)
    return labels, D


def test_nj_textbook_topology():
    labels, D = _textbook_D()
    t = neighbor_joining(labels, D)
    splits = {frozenset(s) for s in t.bipartitions()}
    # canonical Saitou-Nei result groups (A,B) and (D,E)
    assert frozenset({"D", "E"}) in splits
    assert set(t.leaves()) == set(labels)


def test_upgma_ultrametric_and_rf():
    labels, D = _textbook_D()
    tu = upgma(labels, D)
    tj = neighbor_joining(labels, D)
    # RF distance is a non-negative even integer
    rf = rf_distance(tj, tu)
    assert rf >= 0 and rf % 1 == 0


def test_patristic_symmetric_and_zero_diag():
    labels, D = _textbook_D()
    t = neighbor_joining(labels, D)
    leaves, P = patristic_distance_matrix(t)
    assert np.allclose(P, P.T)
    assert np.allclose(np.diag(P), 0.0)
    assert (P[np.triu_indices(len(leaves), 1)] > 0).all()


def test_splits_compatibility():
    L = frozenset("1234")
    assert splits_compatible(frozenset({"3", "4"}), frozenset({"3", "4"}), L)
    # {1,2}|{3,4} vs {1,3}|{2,4} are incompatible (an NNI apart)
    assert not splits_compatible(frozenset({"3", "4"}), frozenset({"2", "4"}), L)


def test_bhv_cone_distance_closed_form():
    L = frozenset("1234")
    a, b = 3.0, 4.0
    p = TreePoint({frozenset({"3", "4"}): a}, L)
    q = TreePoint({frozenset({"2", "4"}): b}, L)
    assert math.isclose(bhv_distance(p, q), a + b, rel_tol=1e-9)

    # add a shared split -> Euclidean contribution
    p2 = TreePoint({frozenset({"3", "4"}): a, frozenset({"2", "3", "4"}): 1.0}, L)
    q2 = TreePoint({frozenset({"2", "4"}): b, frozenset({"2", "3", "4"}): 2.5}, L)
    exp = math.sqrt((a + b) ** 2 + (1.0 - 2.5) ** 2)
    assert math.isclose(bhv_distance(p2, q2), exp, rel_tol=1e-9)


def test_bhv_geodesic_linearity():
    L = frozenset("1234")
    p = TreePoint({frozenset({"3", "4"}): 3.0, frozenset({"2", "3", "4"}): 1.0}, L)
    q = TreePoint({frozenset({"2", "4"}): 4.0, frozenset({"2", "3", "4"}): 2.5}, L)
    D = bhv_distance(p, q)
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        mid = geodesic_interpolate(p, q, t)
        assert math.isclose(bhv_distance(p, mid), t * D, abs_tol=1e-9)


def test_frechet_mean_identity_and_midpoint():
    L = frozenset("1234")
    p = TreePoint({frozenset({"3", "4"}): 3.0}, L)
    same = [p for _ in range(5)]
    m = frechet_mean(same)
    assert frechet_variance(same, m) < 1e-8

    q = TreePoint({frozenset({"2", "4"}): 4.0}, L)
    m2 = frechet_mean([p, q], passes=200)
    # barycentre of two points is (near) equidistant
    assert math.isclose(bhv_distance(m2, p), bhv_distance(m2, q), abs_tol=0.05)


def test_bhv_requires_shared_leafset():
    p = TreePoint({frozenset({"3", "4"}): 1.0}, frozenset("1234"))
    q = TreePoint({frozenset({"3", "4"}): 1.0}, frozenset("12345"))
    with pytest.raises(ValueError):
        bhv_distance(p, q)


def test_distance_from_msa_shapes():
    from refalign.data.synth import SyntheticConfig, generate_corpus
    from refalign.encode import encode_population
    from refalign.msa import center_star_msa
    pops, _ = generate_corpus(SyntheticConfig(n_queries=1, runs_per_query=6, seed=2))
    pop = pops[0]
    alpha, seqs = encode_population(pop)
    labels = [r.run_id for r in pop.runs]
    msa = center_star_msa(labels, [seqs[l] for l in labels], alpha)
    L, D = distance_from_msa(msa)
    assert D.shape == (len(labels), len(labels))
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)
