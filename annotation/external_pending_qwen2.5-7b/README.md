# Unlabeled external annotation package (GSM8K test)

40 failing runs from 24 tasks, each file holding the FIRST successful rollout (the reference annotators see, as in `annotation/annotate.py`) followed by the sampled failing runs. No decisive-step labels exist yet; do not report accuracy on these runs until they are labeled.

Annotate with:

```
python -m annotation.annotate --annotator annotator_XX --traces annotation/external_pending --out annotation/annotations_external
```

Selection: uniform random subset (seed 0) of failing runs whose task has at least one successful rollout.
