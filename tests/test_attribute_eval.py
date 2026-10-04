import numpy as np

from refalign.data.synth import SyntheticConfig, generate_corpus
from refalign.attribute import AttributionConfig, attribute_corpus
from refalign.baselines import run_baseline
from refalign import evaluate as ev


def test_synth_annotations_consistent():
    pops, anns = generate_corpus(SyntheticConfig(n_queries=5, runs_per_query=8, seed=0))
    for pop in pops:
        assert len(pop.successes) >= 1                 # need a healthy lineage
        for f in pop.failures:
            a = anns[f.run_id]
            assert 0 <= a.decisive_step < len(f)       # step is a valid index
            assert a.responsible_agent in set(AGENT_ROSTER)


AGENT_ROSTER = ["Planner", "Researcher", "Coder", "Verifier"]


def test_refalign_beats_random_and_floor():
    pops, anns = generate_corpus(SyntheticConfig(n_queries=30, runs_per_query=10, seed=7))
    preds = attribute_corpus(pops, AttributionConfig())
    rnd = run_baseline("random", pops, seed=7)
    m = ev.score(preds, anns)
    r = ev.score(rnd, anns)
    # RefAlign should clearly beat the random floor on both axes
    assert m.summary()["agent_acc"] > r.summary()["agent_acc"] + 0.15
    assert m.summary()["step_acc"] > r.summary()["step_acc"] + 0.10
    assert m.summary()["agent_acc"] > 0.5


def test_attribution_reproducible():
    pops, anns = generate_corpus(SyntheticConfig(n_queries=10, runs_per_query=8, seed=3))
    p1 = attribute_corpus(pops, AttributionConfig())
    p2 = attribute_corpus(pops, AttributionConfig())
    assert {k: (v.agent, v.step) for k, v in p1.items()} == \
           {k: (v.agent, v.step) for k, v in p2.items()}


def test_all_localizers_and_weightings_run():
    pops, anns = generate_corpus(SyntheticConfig(n_queries=6, runs_per_query=8, seed=1))
    for loc in ["backbone", "sibling", "msa_column"]:
        for w in ["tree", "edit", "uniform"]:
            preds = attribute_corpus(pops, AttributionConfig(localizer=loc, weighting=w))
            assert len(preds) == sum(len(p.failures) for p in pops)
            for pr in preds.values():
                assert pr.agent in AGENT_ROSTER
                assert 0 <= pr.step


def test_bootstrap_ci_bounds_and_paired():
    x = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1], float)
    mean, lo, hi = ev.bootstrap_ci(x, B=2000, seed=0)
    assert lo <= mean <= hi
    assert 0.0 <= lo and hi <= 1.0
    # identical vectors => zero difference, high p
    out = ev.paired_bootstrap(x, x, B=1000, seed=0)
    assert abs(out["diff"]) < 1e-9
    assert out["p_value"] >= 0.5


def test_mcnemar_directional():
    # x correct everywhere y is wrong => significant, small p
    x = np.ones(20, int)
    y = np.zeros(20, int)
    out = ev.mcnemar(x, y)
    assert out["b"] == 20 and out["c"] == 0
    assert out["p_value"] < 1e-4


def test_leaderboard_formatting():
    pops, anns = generate_corpus(SyntheticConfig(n_queries=8, runs_per_query=8, seed=2))
    named = {"phylo": attribute_corpus(pops, AttributionConfig()),
             "last": run_baseline("last_step", pops, seed=2)}
    rows = ev.leaderboard(named, anns, tolerances=(1,), B=500)
    txt = ev.format_leaderboard(rows, tolerances=(1,))
    assert "phylo" in txt and "agent_acc" in txt
