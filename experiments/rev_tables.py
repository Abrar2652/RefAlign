"""LaTeX snippets for the the paper revision from results/revision/*.json.
    PYTHONPATH=. python -m experiments.rev_tables
"""
import json
import os

R = "results/revision"
J = lambda p: json.load(open(p))


def fmt(r):
    return f"{r['c']}/{r['n']} ({r['acc']:.2f})"


def tab_positional():
    d = J(f"{R}/positional_baselines.json")
    S = d["summary"]; T = d["tests_vs_refalign_step"]
    rows = []
    names = [("refalign", "\\pha{} (no fitting)"), ("mode_step", "most frequent step"), ("mode_joint", "most frequent (agent, step)"),
             ("norm_pos", "normalized position"), ("penultimate_step", "penultimate step"), ("last_step", "last step"),
             ("posthoc_const_step3", "constant step 3 (post hoc)")]
    for key, lab in names:
        if key not in S:
            continue
        s = S[key]; t = T.get(key)
        ptxt = (f"{t['b']}/{t['c']} & " + ("$<$0.001" if t['p_holm'] < 0.001 else f"{t['p_holm']:.3f}")) if t else "-- & --"
        rows.append(f"{lab} & {fmt(s['pooled']['step_correct'])} & {fmt(s['pooled']['agent_correct'])} & {fmt(s['pooled']['joint_correct'])} & {ptxt} \\\\")
    tex = r"""% Leakage-controlled positional baselines on the 32 natural failures (leave-one-family-out).
\begin{table}[!t]
\caption{Natural failures ($n{=}32$, four families): positional rules fitted leave-one-family-out beside \pha{}. Counts are correct/$n$ (proportion). $b$/$c$: runs correct only for \pha{} / only for the rule; $p$ is an exact McNemar test on exact step, Holm-corrected over the seven fitted rules (the most-frequent-agent and agent+step rules, omitted here, score 0/32 and 27/32 exact step). Fitted positions beyond a trace are clamped to its last step (never triggered). Family-macro exact step: \pha{} 0.53, most frequent step 0.86, normalized position and penultimate 0.90. The constant step-3 row uses the labels themselves and is a retrospective diagnostic, not a baseline.}
\label{tab:positional}
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}
\toprule
rule & exact step & exact agent & joint & $b$/$c$ & $p_{\mathrm{Holm}}$ \\\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{R}/tab_positional.tex", "w").write(tex)


def tab_intervention():
    rows = []
    for label, path in [("GPT-4o-mini", f"{R}/intervention_rescored_gpt-4o-mini.json"), ("Qwen2.5-7B", f"{R}/intervention_rescored_qwen2.5-7b.json")]:
        d = J(path); P = d["pooled"]; bt, bp = P["baseline_vs_targeted"], P["baseline_vs_placebo"]
        rows.append(f"{label} & {P['baseline']['fail']}/140 & {P['targeted']['fail']}/140 & {P['placebo']['fail']}/140 & "
                    f"{bt['diff']:+.3f} [{bt['ci_lo']:+.3f}, {bt['ci_hi']:+.3f}] & {bt['p_holm']:.3f} & {bp['diff']:+.3f} [{bp['ci_lo']:+.3f}, {bp['ci_hi']:+.3f}] & {bp['p_holm']:.3f} \\\\")
    tex = r"""% Prompt intervention, rescored with the corrected t03 key (76).
\begin{table*}[!t]
\caption{Global prompt intervention, ten task families, $14$ rollouts per family and condition, answer key of the path-area task corrected to $76$ before scoring (model outputs unchanged). Failure counts out of $140$; differences are targeted/placebo minus baseline with Newcombe $95\%$ intervals; $p$ from two-sided Fisher exact tests, Holm-corrected over the two comparisons per model. A global prompt change is not a replay of a single step.}
\label{tab:intervention}
\centering\small
\begin{tabular}{lccccccc}
\toprule
agent model & baseline & Worker instr. & Planner placebo & $\Delta$ Worker [95\% CI] & $p_{\mathrm{Holm}}$ & $\Delta$ Planner [95\% CI] & $p_{\mathrm{Holm}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(f"{R}/tab_intervention_rescored.tex", "w").write(tex)


if __name__ == "__main__":
    os.chdir(".")
    tab_positional(); tab_intervention()
    from experiments.rev_tables_alf import tab_alfworld
    for pre in ["alf_test", "alf_test_complete"]:
        if os.path.exists(f"{R}/{pre}_analysis.json"):
            tab_alfworld(pre); break
    print("tables written")
