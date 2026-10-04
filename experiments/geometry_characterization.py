"""Does the phylogeny organise *failure modes* — the population-level pay-off?

The per-run localisation ablation showed tree-space geometry ~ raw edit distance
(a wash). This experiment tests the fallback the research grounding proposed: a
phylogeny of a population's *failing* runs should carry statistically significant
structure about how they failed.

For each population we build a phylogeny over the failing runs and measure:
  * Blomberg's K and Pagel's lambda phylogenetic signal of the (continuous)
    decisive-step position, with permutation p-values;
  * a permutation test that same-fault-type / same-responsible-agent runs cluster;
  * kNN label purity under tree-space (patristic) vs raw edit distance — the
    honest head-to-head of geometry vs sequence distance for characterisation.

Usage: python -m experiments.geometry_characterization --seeds 0 1 2 3 4
"""
import argparse
import json
import sys
from collections import Counter

import numpy as np

from refalign.data.synth import SyntheticConfig, generate_corpus
from refalign.encode import Alphabet
from refalign.msa import center_star_msa
from refalign.trees import distance_from_msa, upgma, distance_pairwise
from refalign.geometry import patristic_distance_matrix
from refalign.evostat import (blombergs_K, pagels_lambda, continuous_signal_test,
                                knn_purity)


def _majority_purity(labels):
    """Expected kNN purity under random neighbours = sum_c p_c^2 (baseline)."""
    n = len(labels)
    counts = Counter(labels)
    return sum((c / n) ** 2 for c in counts.values())


def run(seeds, n_queries, runs_per_query, k=3):
    Ks, Ls, pK, pL = [], [], [], []
    pur_tree_ft, pur_edit_ft, base_ft = [], [], []
    pur_tree_ag, pur_edit_ag, base_ag = [], [], []
    n_pops = 0

    for seed in seeds:
        cfg = SyntheticConfig(n_queries=n_queries, runs_per_query=runs_per_query,
                              failure_fraction=0.65, regime="algo", seed=seed)
        pops, anns = generate_corpus(cfg)
        for pop in pops:
            fails = pop.failures
            if len(fails) < 5:
                continue
            n_pops += 1
            # phylogeny over failing runs only (UPGMA -> ultrametric -> valid VCV)
            alpha = Alphabet(mode="agent_action")
            seqs = {r.run_id: np.asarray([alpha.encode(s.agent, s.action)
                                          for s in r.steps]) for r in pop.runs}
            labels = [r.run_id for r in fails]
            S = [seqs[l] for l in labels]
            msa = center_star_msa(labels, S, alpha)
            L, D = distance_from_msa(msa)
            tree = upgma(L, D)
            leaves, P = patristic_distance_matrix(tree)

            # traits / labels in tree-leaf order
            step_trait = np.array([anns[l].decisive_step / max(1, len(pop.by_id(l)))
                                   for l in leaves])
            ft = [anns[l].meta.get("fault_type", "?") for l in leaves]
            ag = [anns[l].responsible_agent for l in leaves]

            Ks.append(blombergs_K(P, step_trait))
            Ls.append(pagels_lambda(P, step_trait))
            pK.append(continuous_signal_test(P, step_trait, "K", n_perm=299, seed=seed)["p_value"])
            pL.append(continuous_signal_test(P, step_trait, "lambda", n_perm=299, seed=seed)["p_value"])

            # edit-distance matrix among failing runs, same order as leaves
            _, Ed = distance_pairwise(leaves, [seqs[l] for l in leaves], alpha)

            pur_tree_ft.append(knn_purity(P, ft, k))
            pur_edit_ft.append(knn_purity(Ed, ft, k))
            base_ft.append(_majority_purity(ft))
            pur_tree_ag.append(knn_purity(P, ag, k))
            pur_edit_ag.append(knn_purity(Ed, ag, k))
            base_ag.append(_majority_purity(ag))

    def mean(a):
        a = [x for x in a if np.isfinite(x)]
        return float(np.mean(a)) if a else float("nan")

    def frac_sig(a):
        a = [x for x in a if np.isfinite(x)]
        return float(np.mean([x < 0.05 for x in a])) if a else float("nan")

    return {
        "n_populations": n_pops,
        "blomberg_K_mean": mean(Ks), "K_frac_significant": frac_sig(pK),
        "pagel_lambda_mean": mean(Ls), "lambda_frac_significant": frac_sig(pL),
        "fault_type": {"knn_purity_tree": mean(pur_tree_ft),
                       "knn_purity_edit": mean(pur_edit_ft),
                       "random_baseline": mean(base_ft)},
        "responsible_agent": {"knn_purity_tree": mean(pur_tree_ag),
                              "knn_purity_edit": mean(pur_edit_ag),
                              "random_baseline": mean(base_ag)},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--runs", type=int, default=16)
    ap.add_argument("--out", default="results/geometry_characterization.json")
    args = ap.parse_args(argv)
    res = run(args.seeds, args.queries, args.runs)
    print("=== Failure-mode characterization via the phylogeny ===")
    print(f"populations analysed: {res['n_populations']}")
    print(f"Blomberg's K (decisive-step trait): mean={res['blomberg_K_mean']:.3f}  "
          f"significant(p<.05)={res['K_frac_significant']:.2f}")
    print(f"Pagel's lambda:                     mean={res['pagel_lambda_mean']:.3f}  "
          f"significant(p<.05)={res['lambda_frac_significant']:.2f}")
    for key in ("fault_type", "responsible_agent"):
        d = res[key]
        print(f"kNN purity [{key}]: tree={d['knn_purity_tree']:.3f}  "
              f"edit={d['knn_purity_edit']:.3f}  random={d['random_baseline']:.3f}")
    json.dump(res, open(args.out, "w"), indent=2, default=float)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
