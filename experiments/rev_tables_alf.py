"""LaTeX snippets for the ALFWorld external evaluation (results/revision/alf_test_analysis.json)."""
import json

R = "results/revision"
LAB = {"refalign": "\\pha{} (numeric, unchanged)", "refalign_s": "\\pha{}-S (structured)", "refalign_s_first": "\\pha{}-S, first reference",
       "refalign_s_edit": "\\pha{}-S, action-only nearest", "refalign_s_random": "\\pha{}-S, random reference (5 seeds)",
       "judge_aao": "GPT-4o single-trace (AAO)", "judge_matched": "GPT-4o, \\pha{}-S reference", "judge_first": "GPT-4o, first reference",
       "judge_edit": "GPT-4o, action-only nearest", "judge_random": "GPT-4o, random reference (2 seeds)",
       "lo_mode_step": "LOTO most frequent step", "lo_norm_pos": "LOTO normalized position", "penultimate": "penultimate step",
       "last_step": "last step", "random_step": "random Actor step (5 seeds)"}
ORDER = ["refalign", "refalign_s", "refalign_s_first", "refalign_s_edit", "refalign_s_random", "judge_aao", "judge_matched",
         "judge_first", "judge_edit", "judge_random", "lo_mode_step", "lo_norm_pos", "penultimate", "last_step", "random_step"]


def tab_alfworld(prefix="alf_test_complete"):
    d = json.load(open(f"{R}/{prefix}_analysis.json"))
    f = d["funnel"]; L = d["labels"]
    L.setdefault("n_with_earliest_repairable_strict", 0); L.setdefault("n_with_earliest_repairable_relaxed", 0)
    # funnel table
    bt = f["by_task_type"]
    rows = ["%s & %d & %d & %d & %d \\\\" % (t.replace("_", "\\_"), v["games"], v["runs"], v["success"], v["games_mixed"]) for t, v in bt.items()]
    tex = r"""% ALFWorld collection funnel.
\begin{table}[!t]
\caption{ALFWorld test split (134 unseen games, 8 rollouts each, GPT-4o-mini Planner$\to$Actor, 30-step cap): collection funnel. A failure is reference-eligible when its game has at least one successful rollout; """ + \
    f"{f['reference_eligible_failures']} of {f['failures']} failures are eligible and {f['failures_without_reference']} are not; {f['hit_cap']} failures hit the step cap. " + \
    f"Replay labels: {L['n_profiled']} eligible failures were profiled on steps 1--{L['kmax']}; {L['n_with_earliest_repairable_strict']} have an earliest repairable step under the strict rule ({L['n_with_earliest_repairable_relaxed']} relaxed).}}" + r"""
\label{tab:alf-funnel}
\centering\scriptsize
\begin{tabular}{lcccc}
\toprule
task type & games & rollouts & successes & games with both outcomes \\
\midrule
""" + "\n".join(rows) + f"""
\\midrule
all & {f['games']} & {f['runs']} & {f['successes']} & {f['games_mixed']} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    open(f"{R}/tab_alf_funnel.tex", "w").write(tex)
    # main results table: P1 (strict), replay A, replay B
    P1 = d["P1_exact_step_vs_earliest_repairable"]; A = d["replay_A_expert_at_selected_step"]; B = d["replay_B_repair_at_selected_step"]
    rows = []
    for m in ORDER:
        if m not in P1 and m not in A:
            continue
        p = P1.get(m); a = A.get(m); b = B.get(m)
        c1 = f"{p['c_mean']:.0f}/{p['n']} ({p['acc']:.2f}) [{p['wilson'][0]:.2f}, {p['wilson'][1]:.2f}]" if p else "--"
        c2 = (f"{a['won']}/{a['n_episodes_substitution']} ({a['rate']:.2f})" + (f" vs {a['control_rate']:.2f}" if a.get('control_rate') is not None else "") + (f"; {a['n_runs_step0_selected']} step-0" if a.get('n_runs_step0_selected') else "")) if a and a.get('rate') is not None else "--"
        c3 = f"{b['won']}/{b['n_episodes_substitution']} ({b['rate']:.2f})" if b and b.get('rate') is not None else "--"
        rows.append(f"{LAB[m]} & {c1} & {p['macro']:.2f} & {c2} & {c3} \\\\" if p else f"{LAB[m]} & -- & -- & {c2} & {c3} \\\\")
    T = d["P1_tests"]
    tests = ("none (replay labels pending; see the blocker note)" if not T else "") + "; ".join("%s: $b{=}%d$, $c{=}%d$, $p_{\\mathrm{Holm}}{=}%.3f$" % (k.replace("_vs_", " vs ").replace("_", "\\_"), v["b"], v["c"], v["p_holm"]) for k, v in T.items())
    tex = r"""% ALFWorld attribution and replay results.
\begin{table*}[!t]
\caption{ALFWorld test split: exact-step agreement with the replay-verified earliest repairable step (strict rule; """ + f"$n{{=}}{L['n_with_earliest_repairable_strict']}$ labelled failures; Wilson 95\\% intervals; macro over task types), " + \
    r"""and controlled replay at each method's selected step on the replay subset: A = the environment's expert action is substituted at the selected step and the same policy continues (success rate, with the no-edit resampling control at the same step after ``vs''); B = a fixed repair model proposes the action at the selected step. Replay rates pool two seeds per run; a selected step beyond the run's last action counts as a failed intervention. Paired exact McNemar tests on exact step, Holm over the prespecified family: """ + tests + r"""}
\label{tab:alf-main}
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
method & exact step vs replay label & macro & replay A: expert at selected step (vs no-edit) & replay B: repair model \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(f"{R}/tab_alf_main.tex", "w").write(tex)
    # reference-selection table (label-free form when no replay labels exist)
    RS = d["reference_selection"]; RJ = d["reference_selection_judge"]
    have_labels = any(v.get("n_changed_labelled", 0) for v in RS.values())
    rows = []
    for k, v in RS.items():
        name = k.replace("refalign_s_", "").replace("-s", " seed ")
        bt = v["by_type"]
        bytype = ", ".join("%s %.2f" % (t, x) for t, x in bt.items() if x > 0) or "none"
        if have_labels:
            acc_near = ("%.2f" % v["acc_refalign_s_on_changed"]) if v["acc_refalign_s_on_changed"] is not None else "--"
            acc_alt = ("%.2f" % v["acc_alt_on_changed"]) if v["acc_alt_on_changed"] is not None else "--"
            rows.append("%s & %d/%d (%.3f) & %s & %s (%d) \\\\" % (name, v["n_changed"], v["n"], v["frac_changed"], acc_near, acc_alt, v["n_changed_labelled"]))
        else:
            rows.append("%s & %d/%d (%.3f) & %s \\\\" % (name, v["n_changed"], v["n"], v["frac_changed"], bytype))
    for k, v in RJ.items():
        rows.append("GPT-4o judge: %s & %d/%d & -- \\\\" % (k.replace("judge_", "").replace("-s", " seed "), v["n_changed"], v["n"]))
    pool = ", ".join("%s: %d" % (k, v) for k, v in d["pool_size"].items())
    if have_labels:
        head = "selection rule & changed & nearest acc.\\ on changed & alternative acc.\\ on changed \\\\"; cols = "lccc"
        cap_extra = "and, on the changed and labelled runs, exact-step accuracy of the nearest-reference prediction versus the alternative (count of such runs in parentheses)"
    else:
        head = "selection rule (vs.\\ content-aware nearest) & predicted step changed & changed fraction by task type \\\\"; cols = "lcc"
        cap_extra = "and the fraction changed within each task type (place, examine, clean); replay labels are pending, so accuracy on the changed runs is not reported"
    tex = ("%% Reference selection on ALFWorld.\n\\begin{table}[!t]\n\\caption{Does the reference matter on ALFWorld? For each alternative selection rule applied to \\pha{}-S, the number of the $" + str(next(iter(RS.values()))["n"]) + "$ reference-eligible failures whose predicted step changes relative to the content-aware nearest reference, "
           + cap_extra + ". Pool sizes (successful references: runs): " + pool + ".}\n\\label{tab:alf-refsel}\n\\centering\\scriptsize\n\\begin{tabular}{" + cols + "}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(f"{R}/tab_alf_refsel.tex", "w").write(tex)
    print("alf tables written")


if __name__ == "__main__":
    import os
    os.chdir(".")
    tab_alfworld()
