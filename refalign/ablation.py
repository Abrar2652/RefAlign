"""Ablation suite: the four component ablations from the research grounding
(Section 5), run over multi-seed corpora and pooled for bootstrap CIs and paired
significance tests against the full method.

    (a) alignment          : population MSA  vs  failing-centred pairwise
    (b) tree               : NJ  vs  UPGMA  vs  no-tree (uniform weighting)
    (c) geometry vs distance: tree-space (patristic/BHV)  vs  raw edit  vs  uniform
    (d) evolutionary tests : out-of-repertoire rule  vs  soft support rule
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .types import Annotation, Prediction, RunPopulation
from .data.synth import SyntheticConfig, generate_corpus
from .attribute import AttributionConfig, attribute_corpus
from .baselines import run_baseline

# name -> (ablation group, AttributionConfig)
FULL = AttributionConfig(localizer="backbone", weighting="tree", use_msa="auto",
                         tree_method="nj", divergence_rule="repertoire")

PHYLO_CONFIGS: Dict[str, Tuple[str, AttributionConfig]] = {
    "RefAlign (full)":      ("full", FULL),
    # (a) alignment — adaptive default vs forced strategies
    "align=MSA (forced)":     ("a_alignment",
                               AttributionConfig(localizer="backbone", weighting="tree",
                                                 use_msa=True, tree_method="nj")),
    "align=pairwise (forced)": ("a_alignment",
                               AttributionConfig(localizer="backbone", weighting="tree",
                                                 use_msa=False, tree_method="nj")),
    # (b) tree
    "tree=UPGMA":             ("b_tree",
                               AttributionConfig(localizer="backbone", weighting="tree",
                                                 tree_method="upgma")),
    "-tree (uniform wt)":     ("b_tree",
                               AttributionConfig(localizer="backbone", weighting="uniform",
                                                 tree_method=None)),
    # (c) geometry vs sequence distance  (crux ablation)
    "geom=tree/sibling":      ("c_geometry",
                               AttributionConfig(localizer="sibling", weighting="tree")),
    "geom=edit/sibling":      ("c_geometry",
                               AttributionConfig(localizer="sibling", weighting="edit")),
    "geom=uniform/sibling":   ("c_geometry",
                               AttributionConfig(localizer="sibling", weighting="uniform")),
    # (d) evolutionary/statistical rule
    "rule=support":           ("d_evostat",
                               AttributionConfig(localizer="backbone", weighting="tree",
                                                 divergence_rule="support")),
    # localizer contrast
    "localizer=msa_column":   ("localizer",
                               AttributionConfig(localizer="msa_column", weighting="tree")),
}

DEFAULT_BASELINES = ["random", "first_step", "last_step", "most_active_agent",
                     "longest_agent", "edit_nn"]


@dataclass
class Suite:
    named_preds: Dict[str, Dict[str, Prediction]]
    anns: Dict[str, Annotation]
    groups: Dict[str, str]          # method name -> ablation group
    corpora: List[Tuple[int, List[RunPopulation]]]


def _key(seed: int, run_id: str) -> str:
    return f"s{seed}:{run_id}"


def run_suite(regime: str = "algo", seeds: List[int] = (0, 1, 2),
              n_queries: int = 30, runs_per_query: int = 10,
              baselines: List[str] = None) -> Suite:
    """Run every phylo config + baseline over multiple seeded corpora, pooling
    per-run predictions with seed-namespaced keys for valid paired tests."""
    baselines = list(DEFAULT_BASELINES if baselines is None else baselines)
    named_preds: Dict[str, Dict[str, Prediction]] = {n: {} for n in PHYLO_CONFIGS}
    for b in baselines:
        named_preds[b] = {}
    pooled_anns: Dict[str, Annotation] = {}
    corpora: List[Tuple[int, List[RunPopulation]]] = []

    for seed in seeds:
        cfg = SyntheticConfig(n_queries=n_queries, runs_per_query=runs_per_query,
                              regime=regime, seed=seed)
        pops, anns = generate_corpus(cfg)
        corpora.append((seed, pops))
        for rid, a in anns.items():
            pooled_anns[_key(seed, rid)] = a

        for name, (_grp, acfg) in PHYLO_CONFIGS.items():
            preds = attribute_corpus(pops, acfg)
            for rid, p in preds.items():
                named_preds[name][_key(seed, rid)] = p

        for b in baselines:
            preds = run_baseline(b, pops, seed=seed)
            for rid, p in preds.items():
                named_preds[b][_key(seed, rid)] = p

    groups = {n: g for n, (g, _c) in PHYLO_CONFIGS.items()}
    for b in baselines:
        groups[b] = "baseline"
    return Suite(named_preds, pooled_anns, groups, corpora)
