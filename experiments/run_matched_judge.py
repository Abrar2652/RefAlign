"""Experiment 1: the matched-information LLM judge.

How much of RefAlign's margin over single-trace judges comes from *having* the
successful reference versus from the *alignment algorithm*? The matched judge
(:func:`refalign.llm_baselines.matched_judge`) sees the failing run AND a
successful reference of the same task. Reference variants:

  * ``matched-phylo``   reference = the sibling RefAlign selected
  * ``matched-random``  reference = a uniformly random successful run (seeds 0-4)
  * ``matched-first``   reference = ``successes[0]`` (what the annotators saw;
                         human benchmark only)

The existing single-trace judge (``all_at_once``) and RefAlign (no LLM) run on
the same failing runs, so every comparison is paired. Per-run outcomes go to
``results/new/matched_judge_paired.json``; aggregates, Wilson intervals, exact
McNemar tests (Holm over the four length cells per judge) and token costs go to
``results/new/matched_judge.json``. Every API call is logged to
``results/new/api_log.jsonl`` by :class:`refalign.llm_logged.LoggedLLM`.

    # OpenAI (dated snapshot), when the account has credit:
    python -m experiments.run_matched_judge --judge-name gpt-4o-mini \
        --model gpt-4o-mini-2024-07-18 --workers 4
    # local open model served by vLLM (experiments/serve_local.sh):
    python -m experiments.run_matched_judge --judge-name qwen2.5-7b \
        --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8103/v1 \
        --revision a09a35458c702b33eeacc393d103063234e8bc28
    # rebuild the LaTeX tables from the stored JSON only:
    python -m experiments.run_matched_judge --table-only
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

from refalign.contrastive import ContrastiveConfig, contrastive_localize, _encode_actions, _nearest_sibling
from refalign.llm_baselines import all_at_once, matched_judge, parse_judge_answer, strict_step_parsed
from refalign.llm_logged import LoggedLLM
from refalign import stats_new as S
from refalign.types import Annotation, Prediction, RunPopulation, Trace
from experiments.chain_agents import build_population
from experiments.bench import build_multiculprit, build_diverse
from experiments.run_reference_sensitivity import load_human

CELLS = ["L5", "L12", "L24", "L40", "multiculprit_L12", "reorder_k8", "reorder_k15", "human_n32"]
LENGTH_FAMILY = ["L5", "L12", "L24", "L40"]
CELL_LABEL = {"L5": "$L{=}5$", "L12": "$L{=}12$", "L24": "$L{=}24$", "L40": "$L{=}40$",
              "multiculprit_L12": "multi-culprit", "reorder_k8": "reorder $k{=}8$",
              "reorder_k15": "reorder $k{=}15$", "human_n32": "human"}
# USD per 1M tokens (input, output); local models cost 0.
PRICE = {"gpt-4o-mini-2024-07-18": (0.15, 0.60), "gpt-4o-mini": (0.15, 0.60),
         "gpt-4o-2024-08-06": (2.50, 10.00), "gpt-4o": (2.50, 10.00)}


def det_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


def build_cells(tasks: int = 6, only: Optional[List[str]] = None
                ) -> Dict[str, Tuple[List[RunPopulation], Dict[str, Annotation]]]:
    """The evaluation cells, built exactly as run_contrastive.py / run_gaps.py do."""
    cells: Dict[str, Tuple[List[RunPopulation], Dict[str, Annotation]]] = {}
    want = set(only or CELLS)
    for L in [5, 12, 24, 40]:
        if f"L{L}" not in want:
            continue
        pops, anns = [], {}
        for t in range(tasks):
            p, a = build_population(None, t, L, seed=0, noise=True)
            pops.append(p); anns.update(a)
        cells[f"L{L}"] = (pops, anns)
    if "multiculprit_L12" in want:
        pops, anns = [], {}
        for t in range(tasks):
            p, a = build_multiculprit(t, 12, seed=0, n_fault=8)
            pops.append(p); anns.update(a)
        cells["multiculprit_L12"] = (pops, anns)
    for k in [8, 15]:
        if f"reorder_k{k}" not in want:
            continue
        pops, anns = [], {}
        for t in range(tasks):
            p, a = build_diverse(t, k, seed=0)
            pops.append(p); anns.update(a)
        cells[f"reorder_k{k}"] = (pops, anns)
    if "human_n32" in want:
        cells["human_n32"] = load_human()
    return cells


def phylo_reference(pop: RunPopulation, f: Trace, cfg: ContrastiveConfig = ContrastiveConfig()
                    ) -> Tuple[Trace, Prediction, str]:
    """The reference RefAlign used (``meta['sibling']``); on the fallback path the
    sibling it computed but did not expose, recomputed with the same selector."""
    pred = contrastive_localize(pop, f, cfg)
    sid = pred.meta.get("sibling")
    how = "meta.sibling"
    if sid is None:
        alpha, seqs = _encode_actions(pop)
        sid = _nearest_sibling(f, pop.successes, alpha, seqs, cfg).run_id
        how = "nearest_sibling (fallback path)"
    return pop.by_id(sid), pred, how


def record(pred: Prediction, ann: Annotation, tokens: int, **extra) -> dict:
    return {"agent_pred": pred.agent, "step_pred": int(pred.step),
            "agent_correct": bool(pred.agent == ann.responsible_agent),
            "step_correct": bool(pred.step == ann.decisive_step),
            "tokens": int(tokens), **extra}


def random_reference(pop: RunPopulation, cell: str, run_id: str, seed: int) -> Trace:
    rng = np.random.RandomState(det_seed("matched-random", cell, run_id, seed))
    return pop.successes[int(rng.randint(len(pop.successes)))]


def run_judge(name: str, llm, cells, seeds: List[int], workers: int, paired: dict,
              save_cb=None) -> None:
    """Fill ``paired[name][cell][method][run_id]`` (resumable: done runs are skipped)."""
    J = paired.setdefault(name, {})
    jobs = []
    for cell, (pops, anns) in cells.items():
        P = J.setdefault(cell, {})
        P.setdefault("phylo", {})
        for pop in pops:
            for f in pop.failures:
                if f.run_id not in anns:
                    continue
                ann = anns[f.run_id]
                ref, own, how = phylo_reference(pop, f)
                P["phylo"][f.run_id] = record(own, ann, 0, reference=ref.run_id, reference_how=how,
                                              kind=own.meta.get("kind") or own.meta.get("fallback"))
                jobs.append((cell, "aao", f.run_id, f, None, ann))
                jobs.append((cell, "matched-phylo", f.run_id, f, ref, ann))
                for s in seeds:
                    jobs.append((cell, f"matched-random-s{s}", f.run_id, f,
                                 random_reference(pop, cell, f.run_id, s), ann))
                if cell == "human_n32":
                    jobs.append((cell, "matched-first", f.run_id, f, pop.successes[0], ann))
    todo = [j for j in jobs if j[2] not in J[j[0]].get(j[1], {})]
    print(f"[{name}] {len(jobs)} judge calls planned, {len(todo)} to run", flush=True)

    def work(j):
        cell, method, rid, f, ref, ann = j
        v = llm.view(tag=f"{name}|{cell}|{method}|{rid}")
        if method == "aao":
            p = all_at_once(f, v)
            ag, st, sp, ap_ = parse_judge_answer(v.last_raw, f)
            if sp and not strict_step_parsed(v.last_raw):   # bracketed index: use the tolerant parse
                p = Prediction(p.run_id, ag, st, p.confidence, p.meta)
            r = record(p, ann, sum(v.last_tokens), step_parsed=sp, agent_parsed=ap_,
                       strict_step_parsed=strict_step_parsed(v.last_raw), raw=v.last_raw[:300])
        else:
            p = matched_judge(f, ref, v)
            r = record(p, ann, sum(v.last_tokens), reference=ref.run_id,
                       step_parsed=p.meta["step_parsed"], agent_parsed=p.meta["agent_parsed"],
                       strict_step_parsed=p.meta["strict_step_parsed"], raw=p.meta["raw"][:300])
        return j, r

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, j) for j in todo]
        for fut in as_completed(futs):
            (cell, method, rid, *_), r = fut.result()
            J[cell].setdefault(method, {})[rid] = r
            done += 1
            if done % 200 == 0 or done == len(todo):
                print(f"[{name}] {done}/{len(todo)} calls, {time.time() - t0:.0f}s, "
                      f"{llm.usage.prompt_tokens + llm.usage.completion_tokens} tokens", flush=True)
                if save_cb:
                    save_cb()


def summarize(J: dict, name: str) -> Tuple[dict, dict]:
    """Aggregate one judge's paired outcomes: accuracies with Wilson CIs, token
    costs, parse failures, exact McNemar tests, Holm over the four length cells."""
    summary, tests = {}, {}
    for cell, methods in J.items():
        summary[cell] = {}
        rand_keys = sorted(k for k in methods if k.startswith("matched-random-s"))
        for m, runs in methods.items():
            recs = list(runs.values())
            summary[cell][m] = {
                "step": S.acc_record([r["step_correct"] for r in recs]),
                "agent": S.acc_record([r["agent_correct"] for r in recs]),
                "tokens_mean": float(np.mean([r["tokens"] for r in recs])) if recs else 0.0,
                "tokens_total": int(sum(r["tokens"] for r in recs)),
                "n_unparsed_step": int(sum(1 for r in recs if r.get("step_parsed") is False)),
                "n_unparsed_step_strict": int(sum(1 for r in recs if r.get("strict_step_parsed") is False)),
                "n_reparsed_bracket": int(sum(1 for r in recs if r.get("reparsed_bracket"))),
                "n_unparsed_agent": int(sum(1 for r in recs if r.get("agent_parsed") is False)),
            }
        if rand_keys:
            st = [summary[cell][k]["step"]["acc"] for k in rand_keys]
            ag = [summary[cell][k]["agent"]["acc"] for k in rand_keys]
            summary[cell]["matched-random"] = {
                "seeds": rand_keys,
                "step_mean": float(np.mean(st)), "step_min": float(min(st)), "step_max": float(max(st)),
                "agent_mean": float(np.mean(ag)), "agent_min": float(min(ag)), "agent_max": float(max(ag)),
                "tokens_mean": float(np.mean([summary[cell][k]["tokens_mean"] for k in rand_keys])),
                "n_unparsed_step": int(sum(summary[cell][k]["n_unparsed_step"] for k in rand_keys)),
                "pooled_over_seeds": {
                    "step": S.acc_record([r["step_correct"] for k in rand_keys for r in methods[k].values()]),
                    "agent": S.acc_record([r["agent_correct"] for k in rand_keys for r in methods[k].values()])}}
        tests[cell] = {}
        for comp, (a, b) in {"phylo_vs_matched-phylo": ("phylo", "matched-phylo"),
                             "matched-phylo_vs_aao": ("matched-phylo", "aao"),
                             "phylo_vs_aao": ("phylo", "aao")}.items():
            if a in methods and b in methods:
                tests[cell][comp] = {}
                for metric in ("step", "agent"):
                    x, y, rids = S.paired_vectors(methods[a], methods[b], f"{metric}_correct")
                    tests[cell][comp][metric] = S.mcnemar_exact(x, y)
    # Holm: family = the four length cells, per judge, per comparison, per metric
    for comp in ["phylo_vs_matched-phylo", "matched-phylo_vs_aao", "phylo_vs_aao"]:
        for metric in ("step", "agent"):
            fam = {c: tests[c][comp][metric]["p_value"] for c in LENGTH_FAMILY
                   if c in tests and comp in tests[c]}
            if not fam:
                continue
            adj = S.holm(fam)
            for c, d in adj.items():
                tests[c][comp][metric]["p_holm"] = d["p_holm"]
                tests[c][comp][metric]["holm_family"] = f"{comp}/{metric}: length cells {sorted(fam)} for judge {name} (m={len(fam)})"
                tests[c][comp][metric]["reject_holm_0.05"] = d["reject"]
    return summary, tests


def _cell_val(summary, cell, m):
    d = summary.get(cell, {})
    if m == "matched-random":
        r = d.get("matched-random")
        return f"{r['step_mean']:.2f} [{r['step_min']:.2f}--{r['step_max']:.2f}]" if r else "--"
    r = d.get(m)
    return f"{r['step']['acc']:.2f}" if r else "--"


def write_tex(out: dict, path: str, tests_path: str) -> None:
    judges = list(out["summary"].keys())
    cells = [c for c in CELLS if any(c in out["summary"][j] for j in judges)]
    ns = {c: max((out["summary"][j][c]["phylo"]["step"]["n"] for j in judges if c in out["summary"][j]), default=0)
          for c in cells}
    head = " & ".join(f"{CELL_LABEL[c]} ($n{{=}}{ns[c]}$)" for c in cells)
    lines = [f"\\begin{{tabular}}{{l{'c' * len(cells)}}}", "\\toprule", f"method & {head} \\\\"]
    for j in judges:
        meta = out["judges"][j]
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{len(cells) + 1}}}{{l}}{{\\emph{{judge: {meta['label_tex']}}}}} \\\\")
        for m, label in [("aao", "single-trace AAO"), ("matched-random", "matched-random (mean [min--max], 5 seeds)"),
                         ("matched-phylo", "matched-phylo"), ("matched-first", "matched-first")]:
            vals = [_cell_val(out["summary"][j], c, m) for c in cells]
            if all(v == "--" for v in vals):
                continue
            lines.append(f"{label} & " + " & ".join(vals) + " \\\\")
    lines.append("\\midrule")
    vals = []
    for c in cells:
        v = next((out["summary"][j][c]["phylo"]["step"]["acc"] for j in judges if c in out["summary"][j]), None)
        vals.append(f"{v:.2f}" if v is not None else "--")
    lines.append("\\textbf{RefAlign (no LLM)} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    jd = "; ".join(f"{out['judges'][j]['label_tex']} = \\texttt{{{out['judges'][j]['response_models'][0] if out['judges'][j]['response_models'] else out['judges'][j]['model']}}}"
                   for j in judges)
    tex = ("% Experiment 1: matched-information judge (exact-step accuracy).\n"
           "\\begin{table*}[t]\n\\centering\\small\n" + "\n".join(lines) +
           f"\n\\caption{{Exact-step accuracy of the matched-information judge (sees the failing run and one "
           f"successful reference of the same task) against the single-trace All-at-Once judge and RefAlign, "
           f"on identical failing runs; $n$ per column in the header. Judges run at temperature 0; "
           f"unparseable answers are scored incorrect (counts in \\texttt{{results/new/matched\\_judge.json}}). "
           f"95\\% Wilson intervals for every cell and exact McNemar tests (RefAlign vs.\\ matched-phylo, "
           f"matched-phylo vs.\\ AAO; Holm--Bonferroni over the four length cells per judge and comparison) "
           f"are in Table~\\ref{{tab:matched-judge-tests}}. Judges: {jd}.}}\n"
           "\\label{tab:matched-judge}\n\\end{table*}\n")
    open(path, "w").write(tex)

    # tests table: length cells, per judge
    rows = []
    for j in judges:
        rows.append(f"\\midrule\n\\multicolumn{{9}}{{l}}{{\\emph{{judge: {out['judges'][j]['label_tex']}}}}} \\\\")
        for c in cells:
            T = out["tests"][j].get(c, {})
            s = out["summary"][j][c]
            for comp, lab in [("phylo_vs_matched-phylo", "RefAlign vs.\\ matched-phylo"),
                              ("matched-phylo_vs_aao", "matched-phylo vs.\\ AAO")]:
                if comp not in T:
                    continue
                t = T[comp]["step"]
                a, b = comp.split("_vs_")
                sa, sb = s[a]["step"], s[b]["step"]
                ph = S.fmt_p(t["p_holm"]) if "p_holm" in t else "--"
                rows.append(f"{CELL_LABEL[c]} & {lab} & {S.fmt_acc(sa)} & {S.fmt_acc(sb)} & {t['b']} & {t['c']} & "
                            f"{S.fmt_p(t['p_value'])} & {ph} & {t['cohens_h']:.2f} \\\\")
    tex2 = ("% Experiment 1: paired tests.\n\\begin{table*}[t]\n\\centering\\scriptsize\n"
            "\\begin{tabular}{llccccccc}\n\\toprule\n"
            "cell & comparison (X vs.\\ Y) & acc$_X$ [95\\% Wilson] & acc$_Y$ [95\\% Wilson] & $b$ & $c$ & $p$ & $p_{\\mathrm{Holm}}$ & Cohen's $h$ \\\\\n"
            + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Exact (binomial) McNemar tests on paired exact-step outcomes ($b$: X right, Y wrong; "
            "$c$: X wrong, Y right). $p_{\\mathrm{Holm}}$ applies Holm--Bonferroni within the family of the "
            "four length cells for each judge and comparison ($m{=}4$); the multi-culprit, reorderable-sum and "
            "human cells are single tests and are reported uncorrected. $n$ per cell as in Table~\\ref{tab:matched-judge}.}\n"
            "\\label{tab:matched-judge-tests}\n\\end{table*}\n")
    open(tests_path, "w").write(tex2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-name", default=None, help="short label, e.g. gpt-4o-mini or qwen2.5-7b")
    ap.add_argument("--label-tex", default=None, help="LaTeX label for tables (default: judge name)")
    ap.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (local vLLM); omit for OpenAI")
    ap.add_argument("--revision", default=None, help="HF snapshot hash of a local model (provenance)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--tasks", type=int, default=6)
    ap.add_argument("--out", default="results/new/matched_judge.json")
    ap.add_argument("--paired", default="results/new/matched_judge_paired.json")
    ap.add_argument("--log", default="results/new/api_log.jsonl")
    ap.add_argument("--tex", default="results/new/tab_matched_judge.tex")
    ap.add_argument("--tex-tests", default="results/new/tab_matched_judge_tests.tex")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--merge-from", nargs="*", default=None,
                    help="per-judge paired JSON files to merge into --paired before summarising")
    ap.add_argument("--merge-meta-from", nargs="*", default=None,
                    help="per-judge result JSON files whose 'judges' metadata is merged into --out")
    args = ap.parse_args(argv)

    out = json.load(open(args.out)) if os.path.exists(args.out) else \
        {"design": {}, "judges": {}, "summary": {}, "tests": {}}
    paired = json.load(open(args.paired)) if os.path.exists(args.paired) else {}
    for f in (args.merge_from or []):          # deep union: judge -> cell -> method -> run_id
        for j, cells_ in json.load(open(f)).items():
            for c, methods in cells_.items():
                for m, runs in methods.items():
                    paired.setdefault(j, {}).setdefault(c, {}).setdefault(m, {}).update(runs)
    for f in (args.merge_meta_from or []):
        out["judges"].update(json.load(open(f)).get("judges", {}))
    if args.merge_from:
        json.dump(paired, open(args.paired, "w"), indent=0, default=float)
    out["design"] = {
        "cells": {"L5/L12/L24/L40": "experiments.chain_agents.build_population(None, t, L, seed=0, noise=True), 6 tasks, 4 faults each",
                  "multiculprit_L12": "experiments.bench.build_multiculprit(t, 12, seed=0, n_fault=8), 6 tasks (run_gaps.py settings)",
                  "reorder_k8/k15": "experiments.bench.build_diverse(t, k, seed=0), 6 tasks",
                  "human_n32": "annotation/traces_scale + gold results/human_perrun.json"},
        "methods": {"phylo": "contrastive_localize (default ContrastiveConfig: sibling='content', tau=0.9, top_k=4), no LLM",
                    "aao": "llm_baselines.all_at_once (single-trace)",
                    "matched-phylo": "llm_baselines.matched_judge with reference = RefAlign's selected sibling",
                    "matched-random-s{k}": "matched_judge with a uniformly random successful reference, seed k (det_seed('matched-random', cell, run_id, k))",
                    "matched-first": "matched_judge with reference = pop.successes[0] (human benchmark only)"},
        "scoring": "agent_correct = pred.agent == ann.responsible_agent; step_correct = pred.step == ann.decisive_step (run_contrastive._score); unparseable judge answers -> (<unparsed>, -1) -> incorrect",
        "temperature": 0.0, "seeds_random_reference": args.seeds,
        "stats": "95% Wilson score intervals (scipy binomtest.proportion_ci); exact binomial McNemar (scipy binomtest, cross-checked vs evaluate.mcnemar); Holm-Bonferroni (evaluate.holm_bonferroni) with family = the 4 length cells per judge per comparison; Cohen's h (evaluate.cohens_h)",
    }

    if not args.table_only:
        if not args.judge_name:
            ap.error("--judge-name is required unless --table-only")
        cells = build_cells(args.tasks, args.cells)
        llm = LoggedLLM(args.model, base_url=args.base_url, temperature=0.0,
                        max_tokens=args.max_tokens, log_path=args.log,
                        tag=f"exp1|{args.judge_name}", revision=args.revision)
        seen_models = set()
        _orig = llm.chat_full

        def _chat_full(*a, **k):
            r = _orig(*a, **k)
            seen_models.add(r[3])
            return r
        llm.chat_full = _chat_full
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()

        def save():
            json.dump(paired, open(args.paired, "w"), indent=0, default=float)
        run_judge(args.judge_name, llm, cells, args.seeds, args.workers, paired, save_cb=save)
        save()
        pin, pout = PRICE.get(args.model, (0.0, 0.0))
        usd = (llm.usage.prompt_tokens * pin + llm.usage.completion_tokens * pout) / 1e6
        prev = out["judges"].get(args.judge_name, {})
        out["judges"][args.judge_name] = {
            "model": args.model, "label_tex": args.label_tex or args.judge_name.replace("_", "\\_"),
            "response_models": sorted(seen_models | set(prev.get("response_models", []))),
            "revision": args.revision, "base_url": llm.base_url, "temperature": 0.0,
            "max_tokens": args.max_tokens, "seeds_random_reference": args.seeds,
            "started_utc": prev.get("started_utc", started),
            "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "usage_this_session": {"calls": llm.usage.calls, "prompt_tokens": llm.usage.prompt_tokens,
                                   "completion_tokens": llm.usage.completion_tokens,
                                   "seconds": round(llm.usage.seconds, 1), "errors": llm.n_errors,
                                   "usd": round(usd, 4) if pin else 0.0,
                                   "usd_note": "local model: $0 API cost" if args.base_url else "list price"},
            "usage_note": "cumulative token totals per method are in summary[*][*].tokens_total",
        }
        print(f"[{args.judge_name}] response models seen: {sorted(seen_models)}; "
              f"{llm.usage.calls} calls, {llm.usage.prompt_tokens + llm.usage.completion_tokens} tokens, "
              f"{llm.usage.seconds:.0f}s, ${usd:.3f}", flush=True)

    for j in paired:
        out["summary"][j], out["tests"][j] = summarize(paired[j], j)
        if j not in out["judges"]:
            out["judges"][j] = {"model": j, "label_tex": j, "response_models": []}
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    write_tex(out, args.tex, args.tex_tests)
    for j in out["summary"]:
        print(f"\n=== {j} (exact-step accuracy) ===")
        print(f"  {'cell':18s} {'phylo':>6s} {'aao':>6s} {'m-phylo':>8s} {'m-rand':>8s} {'m-first':>8s} {'unparsed':>9s}")
        for c in CELLS:
            if c not in out["summary"][j] or not {"phylo", "aao", "matched-phylo"} <= set(out["summary"][j][c]):
                continue                      # a partial (per-cell) run: nothing to print for this cell
            s = out["summary"][j][c]
            mr = s.get("matched-random", {}).get("step_mean")
            mf = s.get("matched-first", {}).get("step", {}).get("acc")
            print(f"  {c:18s} {s['phylo']['step']['acc']:6.2f} {s['aao']['step']['acc']:6.2f} "
                  f"{s['matched-phylo']['step']['acc']:8.2f} {(f'{mr:.2f}' if mr is not None else '--'):>8s} "
                  f"{(f'{mf:.2f}' if mf is not None else '--'):>8s} {s['matched-phylo']['n_unparsed_step']:9d}")
    print(f"wrote {args.out}, {args.paired}, {args.tex}, {args.tex_tests}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
