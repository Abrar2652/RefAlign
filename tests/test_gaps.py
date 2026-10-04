"""Tests for the gap-closing capabilities: multi-culprit attribution, all four
fault types (incl. omission via value-aware alignment), and content-aware sibling
selection under solution diversity."""
import numpy as np

from experiments.bench import build_multiculprit, build_diverse, FAULT_TYPES
from refalign.contrastive import contrastive_localize, ContrastiveConfig


def _score(pops, anns, cfg=None):
    preds = {}
    for p in pops:
        for f in p.failures:
            preds[f.run_id] = contrastive_localize(p, f, cfg or ContrastiveConfig())
    rids = [r for r in preds if r in anns]
    agent = np.mean([preds[r].agent == anns[r].responsible_agent for r in rids])
    step = np.mean([preds[r].step == anns[r].decisive_step for r in rids])
    return agent, step, preds, rids


def test_multiculprit_agent_and_step():
    pops, anns = [], {}
    for t in range(6):
        p, a = build_multiculprit(t, 12, seed=0, n_fault=8)
        pops.append(p); anns.update(a)
    agent, step, _, rids = _score(pops, anns)
    # 4 rotating worker roles => agent chance ~0.17; method must be far above
    assert agent > 0.85, agent
    assert step > 0.85, step


def test_all_fault_types_localized():
    pops, anns = [], {}
    for t in range(6):
        p, a = build_multiculprit(t, 12, seed=1, n_fault=8)
        pops.append(p); anns.update(a)
    _, _, preds, rids = _score(pops, anns)
    from collections import defaultdict
    byt = defaultdict(list)
    for r in rids:
        byt[anns[r].meta["fault_type"]].append(r)
    assert set(byt) == set(FAULT_TYPES)          # all four types present
    for ftype, rs in byt.items():
        acc = np.mean([preds[r].step == anns[r].decisive_step for r in rs])
        assert acc >= 0.75, f"{ftype}: {acc}"     # incl. 'skip' (value-aware alignment)


def test_content_aware_beats_edit_under_diversity():
    pops, anns = [], {}
    for t in range(6):
        p, a = build_diverse(t, 10, seed=0); pops.append(p); anns.update(a)
    _, edit_step, _, _ = _score(pops, anns, ContrastiveConfig(sibling="edit"))
    _, content_step, _, _ = _score(pops, anns, ContrastiveConfig(sibling="content"))
    # legitimate solution diversity breaks naive edit selection; content-aware fixes it
    assert content_step > 0.8
    assert content_step > edit_step + 0.3
