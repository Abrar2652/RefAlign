"""RefAlign failure-attribution method.

Premise: a failing run is one lineage that *diverged* from a population of
healthy (successful) runs of the same query. Aligning the population and
inferring a phylogeny lets us (i) weight each healthy run by its tree-space
proximity to the failing run and (ii) reconstruct, column by column, what the
healthy lineage reliably does. The decisive error is the **earliest conserved
column at which the failing run leaves the healthy repertoire**.

Concretely, for a failing run f and the successful runs S:

  1. build a population MSA and (from its distances) a phylogeny;
  2. weight each s in S by proximity to f  (``weighting``):
        'tree'    -> inverse patristic distance on the phylogeny  (geometry)
        'edit'    -> inverse raw Needleman-Wunsch distance        (no geometry)
        'uniform' -> equal weight                                 (no geometry)
  3. per alignment column, compute the weighted healthy support of each symbol;
  4. the decisive column is the earliest one that is *conserved* among the
     healthy runs (a dominant symbol with support >= ``conservation_threshold``)
     yet the failing run's symbol has support < ``support_floor``;
  5. map that column back to a (step, agent) in the failing run.

The four ablation knobs (alignment / tree / geometry-vs-sequence / evo-stats)
are all exposed on :class:`AttributionConfig`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .types import Trace, Prediction, RunPopulation
from .encode import Alphabet, GAP, encode_population
from .msa import center_star_msa, MultipleAlignment, needleman_wunsch
from .trees import build_tree, distance_from_msa
from .geometry import patristic_distance_matrix

EPS = 1e-9


@dataclass
class AttributionConfig:
    # (a) alignment: True = population MSA (best for short traces), False =
    # failing-centred pairwise star (best for long/repetitive traces), 'auto' =
    # pick by median trace length (the recommended, regime-adaptive default).
    use_msa: Union[bool, str] = "auto"
    auto_msa_max_len: int = 15           # median length above which 'auto' -> pairwise
    tree_method: Optional[str] = "nj"    # (b) 'nj' | 'upgma' | None (flat, no tree)
    weighting: str = "tree"              # (c) 'tree' | 'edit' | 'uniform'
    localizer: str = "backbone"          # 'backbone' (re-align to consensus) | 'msa_column'
    conservation_threshold: float = 0.5  # (d) a healthy column needs a dominant symbol
    divergence_rule: str = "repertoire"  # 'repertoire' (out-of-repertoire) | 'support'
    support_floor: float = 0.15          # (used only when divergence_rule == 'support')
    distance_metric: str = "align"
    encode_mode: str = "agent_action"


# ---------------------------------------------------------------------------
def _agents_per_column(run: Trace, row: np.ndarray) -> List[Optional[str]]:
    """For each MSA column, the agent acting in ``run`` (None on a gap)."""
    out: List[Optional[str]] = []
    step = 0
    for x in row.tolist():
        if x == GAP:
            out.append(None)
        else:
            out.append(run.steps[step].agent)
            step += 1
    return out


def _col_to_step(row: np.ndarray, c: int) -> int:
    """Failing-run step index corresponding to column ``c`` (non-gap count before c)."""
    return int(np.sum(row[:c] != GAP))


@dataclass
class _PopArtifacts:
    alphabet: Alphabet
    seqs: Dict[str, np.ndarray]
    msa: MultipleAlignment
    patristic: Optional[Tuple[List[str], np.ndarray]]


def _resolve_use_msa(pop: RunPopulation, cfg: AttributionConfig) -> bool:
    """Resolve the (possibly 'auto') alignment strategy for a population.

    'auto' uses the population median trace length: short traces favour a
    global population MSA, long/repetitive traces favour a failing-centred
    pairwise star alignment (see ablation (a)).
    """
    if cfg.use_msa in (True, False):
        return bool(cfg.use_msa)
    lengths = sorted(len(r) for r in pop.runs)
    median = lengths[len(lengths) // 2] if lengths else 0
    return median <= cfg.auto_msa_max_len


def _build_artifacts(pop: RunPopulation, cfg: AttributionConfig,
                     center_label: Optional[str] = None) -> _PopArtifacts:
    alphabet, seqs = encode_population(pop, mode=cfg.encode_mode)
    labels = [r.run_id for r in pop.runs]
    S = [seqs[l] for l in labels]
    center = labels.index(center_label) if center_label is not None else None
    msa = center_star_msa(labels, S, alphabet, center=center)
    patristic = None
    if cfg.tree_method is not None:
        L, D = distance_from_msa(msa, metric=cfg.distance_metric)
        tree = build_tree(L, D, method=cfg.tree_method)
        patristic = patristic_distance_matrix(tree)
    return _PopArtifacts(alphabet, seqs, msa, patristic)


def _proximity_weights(pop: RunPopulation, failing: Trace, success_ids: List[str],
                       art: _PopArtifacts, cfg: AttributionConfig) -> Dict[str, float]:
    """Weight each successful run by proximity to the failing run."""
    if cfg.weighting == "uniform" or not success_ids:
        return {sid: 1.0 for sid in success_ids}

    dist: Dict[str, float] = {}
    if cfg.weighting == "tree" and art.patristic is not None:
        labels, D = art.patristic
        fi = labels.index(failing.run_id)
        for sid in success_ids:
            dist[sid] = float(D[fi, labels.index(sid)])
    elif cfg.weighting == "edit" or art.patristic is None:
        fa = art.seqs[failing.run_id]
        for sid in success_ids:
            cost, _, _ = needleman_wunsch(fa, art.seqs[sid], art.alphabet)
            denom = max(len(fa), len(art.seqs[sid]), 1)
            dist[sid] = cost / denom
    else:  # tree requested but unavailable
        return {sid: 1.0 for sid in success_ids}

    # inverse-distance weighting (nearest healthy relatives dominate)
    return {sid: 1.0 / (dist[sid] + EPS) for sid in success_ids}


def _weighted_column_support(msa: MultipleAlignment, success_ids: List[str],
                             weights: Dict[str, float]
                             ) -> List[Dict[int, float]]:
    """Per column: {symbol_id -> weighted fraction of healthy support} (gap = GAP id)."""
    ncol = msa.n_cols
    supports: List[Dict[int, float]] = [defaultdict(float) for _ in range(ncol)]
    wsum = sum(weights.get(s, 0.0) for s in success_ids) or 1.0
    for sid in success_ids:
        w = weights.get(sid, 0.0)
        if w <= 0:
            continue
        row = msa.row(sid)
        for c in range(ncol):
            supports[c][int(row[c])] += w / wsum
    return [dict(s) for s in supports]


def _raw_column_counts(msa: MultipleAlignment, success_ids: List[str]
                       ) -> List[Dict[int, int]]:
    """Per column: {symbol_id -> unweighted number of healthy runs using it}."""
    ncol = msa.n_cols
    counts: List[Dict[int, int]] = [defaultdict(int) for _ in range(ncol)]
    for sid in success_ids:
        row = msa.row(sid)
        for c in range(ncol):
            counts[c][int(row[c])] += 1
    return [dict(c) for c in counts]


def _weighted_modal_agent(msa: MultipleAlignment, pop: RunPopulation,
                          success_ids: List[str], weights: Dict[str, float],
                          col: int) -> Optional[str]:
    tally: Dict[str, float] = defaultdict(float)
    for sid in success_ids:
        run = pop.by_id(sid)
        ag = _agents_per_column(run, msa.row(sid))[col]
        if ag is not None:
            tally[ag] += weights.get(sid, 0.0)
    if not tally:
        return None
    return max(tally.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
def attribute_run(pop: RunPopulation, failing: Trace, cfg: AttributionConfig,
                  art: Optional[_PopArtifacts] = None) -> Prediction:
    """Attribute a single failing run within its population."""
    success_ids = [r.run_id for r in pop.successes]
    if art is None:
        center = None if _resolve_use_msa(pop, cfg) else failing.run_id
        art = _build_artifacts(pop, cfg, center_label=center)

    # no healthy runs to compare against -> fall back to last step
    if not success_ids:
        last = len(failing) - 1
        return Prediction(failing.run_id, failing.agent_at(last), last, 0.0,
                          meta={"fallback": "no_successes"})

    weights = _proximity_weights(pop, failing, success_ids, art, cfg)
    supports = _weighted_column_support(art.msa, success_ids, weights)
    raw_counts = _raw_column_counts(art.msa, success_ids)

    if cfg.localizer == "backbone":
        pred = _localize_backbone(pop, failing, art, cfg, success_ids,
                                  weights, supports, raw_counts)
    elif cfg.localizer == "sibling":
        pred = _localize_sibling(pop, failing, art, cfg, success_ids,
                                 weights, supports, raw_counts)
    else:
        pred = _localize_msa_column(pop, failing, art, cfg, success_ids,
                                    weights, supports, raw_counts)
    pred.meta.update({"weighting": cfg.weighting, "use_msa": cfg.use_msa,
                      "tree_method": cfg.tree_method, "localizer": cfg.localizer})
    return pred


def _consensus_backbone(art: _PopArtifacts, success_ids: List[str],
                        supports, raw_counts, cfg: AttributionConfig,
                        pop: RunPopulation, weights):
    """Recover the conserved backbone: ordered (symbol, agent, msa_col) for every
    column whose dominant *non-gap* symbol is conserved among healthy runs."""
    backbone = []
    for c in range(art.msa.n_cols):
        non_gap = {s: v for s, v in supports[c].items() if s != GAP}
        if not non_gap:
            continue
        dom_sym, dom_supp = max(non_gap.items(), key=lambda kv: kv[1])
        if dom_supp < cfg.conservation_threshold:
            continue
        agent = _weighted_modal_agent(art.msa, pop, success_ids, weights, c)
        backbone.append((int(dom_sym), agent, c))
    return backbone


def _localize_backbone(pop, failing, art, cfg, success_ids, weights,
                       supports, raw_counts) -> Prediction:
    """Re-align the failing run to the recovered conserved backbone.

    Robust to optional-step indels and long/repetitive traces where the global
    MSA row of a derailed run is unreliable: the backbone is short and clean, so
    a fresh pairwise alignment pins each failing step to its intended phase.
    """
    backbone = _consensus_backbone(art, success_ids, supports, raw_counts, cfg,
                                   pop, weights)
    if not backbone:
        last = len(failing) - 1
        return Prediction(failing.run_id, failing.agent_at(last), last, 0.0,
                          meta={"fallback": "no_backbone"})

    fseq = art.seqs[failing.run_id]
    cseq = np.asarray([b[0] for b in backbone], dtype=np.int64)
    _, fa, ca = needleman_wunsch(fseq, cseq, art.alphabet)

    fp = 0   # failing step pointer
    cp = 0   # backbone pointer
    for f, c in zip(fa.tolist(), ca.tolist()):
        if c != GAP:
            _sym, bagent, bcol = backbone[cp]
            if f == GAP:
                step = min(fp, len(failing) - 1)
                agent = bagent or failing.agent_at(step)
                return Prediction(failing.run_id, agent, step, 1.0,
                                  meta={"kind": "deletion", "backbone_col": bcol})
            if f != _sym and raw_counts[bcol].get(int(f), 0) == 0:
                step = min(fp, len(failing) - 1)
                return Prediction(failing.run_id, failing.agent_at(step), step, 1.0,
                                  meta={"kind": "substitution", "backbone_col": bcol})
            cp += 1
        if f != GAP:
            fp += 1

    last = len(failing) - 1
    return Prediction(failing.run_id, failing.agent_at(last), last, 0.0,
                      meta={"fallback": "no_divergence"})


def _localize_sibling(pop, failing, art, cfg, success_ids, weights,
                      supports, raw_counts) -> Prediction:
    """Anchor localisation on the nearest healthy *sibling* (the counterfactual).

    The sibling shares the failing run's optional-step pattern and cycle path, so
    aligning failing<->sibling disambiguates *which* occurrence of a repeated
    action the fault falls on — the exact failure of the averaged backbone on
    long, repetitive traces. The sibling is chosen by ``weighting`` (tree-space
    patristic distance vs raw edit distance), so this is where the geometry-vs-
    sequence-distance comparison (crux ablation c) actually bites.
    """
    backbone = _consensus_backbone(art, success_ids, supports, raw_counts, cfg,
                                   pop, weights)
    if not backbone:
        last = len(failing) - 1
        return Prediction(failing.run_id, failing.agent_at(last), last, 0.0,
                          meta={"fallback": "no_backbone"})

    # nearest sibling = highest proximity weight (weights are inverse-distance)
    s_star = max(success_ids, key=lambda s: weights.get(s, 0.0))
    sseq = art.seqs[s_star]
    cseq = np.asarray([b[0] for b in backbone], dtype=np.int64)

    # map each sibling step to a backbone index (sibling is clean healthy)
    _, sa, ca = needleman_wunsch(sseq, cseq, art.alphabet)
    sib_to_bidx: Dict[int, int] = {}
    sp = cp = 0
    for s, c in zip(sa.tolist(), ca.tolist()):
        if s != GAP and c != GAP:
            sib_to_bidx[sp] = cp
        if c != GAP:
            cp += 1
        if s != GAP:
            sp += 1

    # align failing to the sibling, evaluate divergence only at backbone steps
    fseq = art.seqs[failing.run_id]
    _, fa, ga = needleman_wunsch(fseq, sseq, art.alphabet)
    fp = gp = 0
    for f, g in zip(fa.tolist(), ga.tolist()):
        if g != GAP:
            bidx = sib_to_bidx.get(gp)
            if bidx is not None:
                sym, bagent, bcol = backbone[bidx]
                if f == GAP:
                    step = min(fp, len(failing) - 1)
                    agent = bagent or failing.agent_at(step)
                    return Prediction(failing.run_id, agent, step, 1.0,
                                      meta={"kind": "deletion", "backbone_col": bcol,
                                            "sibling": s_star})
                if f != sym and raw_counts[bcol].get(int(f), 0) == 0:
                    step = min(fp, len(failing) - 1)
                    return Prediction(failing.run_id, failing.agent_at(step), step, 1.0,
                                      meta={"kind": "substitution", "backbone_col": bcol,
                                            "sibling": s_star})
            gp += 1
        if f != GAP:
            fp += 1

    last = len(failing) - 1
    return Prediction(failing.run_id, failing.agent_at(last), last, 0.0,
                      meta={"fallback": "no_divergence", "sibling": s_star})


def _localize_msa_column(pop, failing, art, cfg, success_ids, weights,
                         supports, raw_counts) -> Prediction:
    """Original localizer: scan the failing run's own MSA row for the earliest
    conserved divergence. Kept as an ablation of the backbone re-alignment."""
    frow = art.msa.row(failing.run_id)
    for c in range(art.msa.n_cols):
        non_gap = {sym: v for sym, v in supports[c].items() if sym != GAP}
        if not non_gap:
            continue
        _max_sym, max_supp = max(non_gap.items(), key=lambda kv: kv[1])
        if max_supp < cfg.conservation_threshold:
            continue
        fs = int(frow[c])
        if fs == GAP:
            divergent, fs_supp = True, 0.0
        elif cfg.divergence_rule == "repertoire":
            fs_supp = supports[c].get(fs, 0.0)
            divergent = raw_counts[c].get(fs, 0) == 0
        else:
            fs_supp = supports[c].get(fs, 0.0)
            divergent = fs_supp < cfg.support_floor
        if not divergent:
            continue
        if fs != GAP:
            step = min(_col_to_step(frow, c), len(failing) - 1)
            return Prediction(failing.run_id, failing.agent_at(step), step,
                              float(max(0.0, max_supp - fs_supp)),
                              meta={"kind": "substitution", "decisive_col": c})
        step = min(_col_to_step(frow, c), len(failing) - 1)
        agent = _weighted_modal_agent(art.msa, pop, success_ids, weights, c) \
            or failing.agent_at(step)
        return Prediction(failing.run_id, agent, step, float(max(0.0, max_supp)),
                          meta={"kind": "deletion", "decisive_col": c})

    last = len(failing) - 1
    return Prediction(failing.run_id, failing.agent_at(last), last, 0.0,
                      meta={"fallback": "no_divergence"})


def attribute_population(pop: RunPopulation, cfg: AttributionConfig
                         ) -> Dict[str, Prediction]:
    """Attribute every failing run in a population (artifacts built once when possible)."""
    preds: Dict[str, Prediction] = {}
    use_msa = _resolve_use_msa(pop, cfg)
    # a population MSA is shared across failing runs; a failing-centred pairwise
    # alignment must be rebuilt per run.
    shared_art = _build_artifacts(pop, cfg) if use_msa else None
    for failing in pop.failures:
        art = shared_art if use_msa else _build_artifacts(
            pop, cfg, center_label=failing.run_id)
        preds[failing.run_id] = attribute_run(pop, failing, cfg, art=art)
    return preds


def attribute_corpus(pops: List[RunPopulation], cfg: AttributionConfig
                     ) -> Dict[str, Prediction]:
    out: Dict[str, Prediction] = {}
    for pop in pops:
        out.update(attribute_population(pop, cfg))
    return out
