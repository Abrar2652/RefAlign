import numpy as np

from refalign.data.synth import SyntheticConfig, generate_corpus
from refalign.encode import Alphabet, GAP, encode_population
from refalign.msa import needleman_wunsch, center_star_msa, pairwise_cost


def test_alphabet_substitution_cost():
    a = Alphabet(mode="agent_action")
    code = a.encode("Coder", "code")
    run = a.encode("Coder", "run")
    search = a.encode("Researcher", "search")
    assert a.substitution_cost(code, code) == 0.0
    assert a.substitution_cost(code, run) == 0.5      # same agent, different action
    assert a.substitution_cost(code, search) == 1.0   # different agent
    assert a.substitution_cost(code, GAP) == 1.0


def test_needleman_wunsch_identity_and_gaps():
    a = Alphabet(mode="agent_action")
    s = np.array([a.encode("Planner", "plan"), a.encode("Coder", "code")])
    cost, aa, bb = needleman_wunsch(s, s, a)
    assert cost == 0.0
    assert np.array_equal(aa, bb)

    # one deletion => aligned length grows, cost == 1 gap
    t = np.array([s[0], a.encode("Researcher", "search"), s[1]])
    cost, aa, bb = needleman_wunsch(s, t, a)
    assert cost == 1.0
    assert len(aa) == len(bb) == 3
    assert (aa == GAP).sum() == 1


def test_msa_gap_removal_invariant():
    pops, _ = generate_corpus(SyntheticConfig(n_queries=2, runs_per_query=8, seed=1))
    pop = pops[0]
    alpha, seqs = encode_population(pop)
    labels = [r.run_id for r in pop.runs]
    msa = center_star_msa(labels, [seqs[l] for l in labels], alpha)
    # removing gaps from any aligned row recovers the original sequence
    for l in labels:
        row = msa.row(l)
        assert np.array_equal(row[row != GAP], seqs[l])
    # rectangular matrix
    assert msa.matrix.shape[0] == len(labels)
    assert all(len(r) == msa.n_cols for r in msa.matrix)


def test_pairwise_cost_symmetric():
    a = Alphabet(mode="agent_action")
    x = np.array([a.encode("Planner", "plan"), a.encode("Coder", "code")])
    y = np.array([a.encode("Planner", "plan"), a.encode("Coder", "run")])
    assert pairwise_cost(x, y, a) == pairwise_cost(y, x, a)
