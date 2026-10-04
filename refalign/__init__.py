"""RefAlign — contrastive counterfactual failure attribution for multi-agent
LLM systems.

The central idea: a *single* multi-agent run is one trajectory, but many runs of
the same task form a *population of homologous sequences*. Treating them with the
tools of comparative sequence analysis — aligning runs and contrasting a failing
run against its successful siblings — localises the decisive error (the earliest
action whose correction would flip failure to success) far better than reading a
single trace, especially on long runs, and at no LLM cost.

Core method:

    contrastive   align a failing run to its nearest successful sibling; return
                  the earliest step diverging in action OR content value.

Supporting modules:

    encode/msa    the (agent, action) alphabet and multiple/pairwise alignment
    evaluate      Who&When metrics (agent / step / step +-k), bootstrap CIs,
                  paired tests, effect sizes, Holm correction
    baselines     heuristic + edit-NN attributors
    llm/llm_baselines   Who&When LLM baselines (All-at-Once / Step / Binary)

The `trees`, `geometry`, and `evostat` modules implement a tree-space / phylogenetic
alternative that was evaluated and did *not* improve attribution (see RESULTS.md
"design ablations"); they are retained as ablation baselines, not the method.
"""

from .types import Step, Trace, Annotation, Prediction, RunPopulation
from .contrastive import (contrastive_localize, contrastive_llm,
                          attribute_corpus_contrastive, ContrastiveConfig)

__all__ = ["Step", "Trace", "Annotation", "Prediction", "RunPopulation",
           "contrastive_localize", "contrastive_llm",
           "attribute_corpus_contrastive", "ContrastiveConfig"]
__version__ = "1.0.0"
