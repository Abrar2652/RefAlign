# Pre-specified design: external evaluation on ALFWorld household tasks (written before any outcome was inspected; 2026-10-01)

## Why this benchmark
ALFWorld (Shridhar et al., ICLR 2021) is an established text-based embodied benchmark whose original tasks and agent execution are reproducible offline: each game is a fixed initial state with a deterministic simulator, success is decided by the environment's own goal check (independent of the agent), every intermediate action and observation is observable, and the behavior is non-numeric, stateful tool use (navigate, open, take, heat, cool, clean, put). The same game can be rolled out repeatedly to obtain successful and failed runs of the same task, and the simulator exposes an expert planner that supplies a known valid action from any state, which permits controlled replay. Who&When, MAST and TRAIL were checked and rejected as sources of same-task successful references: their logs are single runs of web/tool tasks whose environments (live web pages, GAIA files, the original agent frameworks and model snapshots) cannot be re-executed here.

## Task sampling
- Development games: 40 games drawn with `RandomState(0)` from `valid_seen` (in-distribution rooms). Used only to debug the pipeline, choose prompts, and develop any detector extension. No test-set outcome is examined before the method is frozen.
- Test games: all games of `valid_unseen` (the standard out-of-distribution evaluation split, 134 games, six task types). If budget forces a subset, games are drawn with `RandomState(1)` stratified by task type, and the subset size is recorded before collection.
- Task family = task type (pick_and_place_simple, look_at_obj_in_light, pick_clean_then_place_in_recep, pick_heat_then_place_in_recep, pick_cool_then_place_in_recep, pick_two_obj_and_place). Problem template = the game's goal template. Rollouts of the same game never cross a split.

## Agent system, models, sampling
- Planner (one call) -> Actor (one call per step choosing one admissible command) -> environment observation, as in `experiments/alfworld_agents.py`. Admissible commands are shown to the Actor. Invalid outputs map to the closest admissible command (difflib ratio >= 0.6) or `look`; the count is recorded.
- Policy model: `gpt-4o-mini-2024-07-18`, temperature 1.0, max 160 output tokens per call, one fixed seed per call derived from (game, rollout, step) so every rollout is reproducible modulo provider nondeterminism; the response model id and fingerprint are logged per call.
- Rollouts per game: 10 (seeds 0-9). Maximum 30 Actor steps per rollout; hitting the cap is a failure and is flagged.
- Success verification: the environment's goal predicate (`info['won']`).

## Reference pool and eligibility
- For each failing run, the reference pool is the set of successful rollouts of the same game. Reference-eligible failures = failing runs of games with >= 1 success. Failures of games with no success are kept in the coverage analysis and never attributed by reference-based methods (end-to-end performance counts them as unattributed).
- Collection funnel to report: games sampled -> rollouts generated -> successful/failed -> reference-eligible failures -> replay-labeled failures.

## Labels (no human annotation is claimed)
- Replay-verified step profile (primary labels): for a failing run f and an Actor step k, the prefix a_1..a_{k-1} is re-executed, the expert action from the resulting state replaces a_k, and the same policy continues from step k+1 with fresh seeds (R = 2 replay seeds). p_k = fraction of successful replays. The *earliest repairable step* is the smallest k with p_k >= 0.5 (both replays succeed when R = 2); runs where no k <= K_max qualifies are "unrepairable within budget" and reported, not dropped. K_max = min(number of Actor steps, 12) to bound cost; steps beyond K_max are not profiled and this truncation is reported.
- No-edit control: same prefix re-execution at step k with resampling and no override, R = 2 seeds, to measure recovery from resampling alone.
- Because any single replayed step supports only a local-repair claim, we report the whole profile where available and use the earliest repairable step as a label while stating that multiple repair points may exist.

## Methods compared (identical reference pools and failing runs)
1. RefAlign (unchanged numeric implementation; expected to fire on receptacle indices or fall back).
2. RefAlign-S: the same alignment with a structured comparator on tool arguments (verb, object, receptacle) at aligned backbone steps, developed on the development games only and frozen before test evaluation; reported as a task-specific comparator variant.
3. Reference-conditioned judge: `gpt-4o-2024-08-06`, temperature 0, same prompt as `matched_judge`, reference = RefAlign-S's selected reference; variants with first, random (5 seeds), action-only-nearest and content-aware-nearest references hold the judge fixed.
4. Single-trace judge (AAO), same model.
5. Positional baselines, leave-one-task-type-out: mode step, normalized position, penultimate, last.
6. Random eligible step (uniform over Actor steps, 5 seeds).
7. Oracle location: the earliest repairable step from the replay profile (upper-bound diagnostic, clearly labeled).

## Replay experiments
- A. Localization validity: at each method's selected step, apply the expert action and continue with the policy (R = 2 seeds); outcome = success rate; compared with the no-edit control at the same step.
- B. Practical repair: at each method's selected step, a repair model (`gpt-4o-mini-2024-07-18`, temperature 0, one call, sees only the prefix, the goal and the admissible commands, and is told the step is suspected wrong) proposes a replacement action; the policy continues (R = 2 seeds). Same repair prompt and budget for every method.
- Reported: repair success rate, improvement over no-edit control (paired by run and seed), cost (LLM calls and tokens), and the fraction of selected steps where no override was possible (e.g., selected step beyond the run's length).

## Primary metrics and comparisons (prespecified)
- P1: exact-step agreement with the earliest repairable step, on profiled failures (counts / denominators, pooled and task-type macro).
- P2: replay-A success at the selected step minus no-edit control (paired over runs and seeds).
- Primary family for Holm correction (m = 4): RefAlign-S vs reference-conditioned judge; RefAlign-S vs strongest leakage-controlled positional baseline; RefAlign-S vs random step; RefAlign-S vs RefAlign, all on P1 (exact McNemar on paired runs). P2 comparisons use the same family. Uncertainty: 95% Wilson per cell plus a task-type-level cluster bootstrap (resampling games) where >= 6 clusters exist; with six task types, per-type results are shown and no population-level claim is made beyond them.
- Secondary: exact-agent, joint, within-one-step (reported, not emphasized), reference coverage, fallback/abstention rate, conditional (reference-eligible) vs end-to-end (all failures) performance, acquisition cost (rollouts and tokens to obtain a reference) separately from attribution cost.

## Budget and stopping rule
Pilot on development games with a local open model first (free). Paid collection proceeds only if the pilot shows a success rate between 0.15 and 0.90 on development games (so test populations will contain both outcomes); otherwise the max-step budget or the policy model is revised before any test rollout. Expected paid volume is recorded in the results file before launch and the actual spend afterwards.

## Amendments made on the development split only (recorded before any test-split rollout)
1. **Observation truncation (bug).** The first pilot truncated the Actor's *current* observation to 300 characters, which could hide the target object in long receptacle listings. The current observation is now shown in full (up to 1,500 characters); history entries stay at 300 characters, and trace contents keep 300 characters.
2. **Verb vocabulary.** The installed ALFWorld (0.4.2 / TextWorld 1.7) phrases placing as `move X to Y`; `move` was added to the action-label vocabulary (it was mapped to `other`).
3. **Few-shot exemplars.** The first pilot (12 development games x 2 rollouts, gpt-4o-mini, no exemplars) succeeded on 7/24 episodes, 6 of them pick-and-place, with 17/24 hitting the 30-step cap by looping; only 3 failures had a same-game success. Following the benchmark's standard protocol, the Actor prompt now includes the two public act-only exemplars of the matching task type from the ReAct prompt file (Yao et al., 2023). The repair model in replay mode B receives the same exemplars. This is a change to the agent system, not to any attribution method.
4. **Budget cap.** New paid inference for this revision is capped at about USD 40 at list prices (collection, judges, replays); if the pre-specified design exceeds it, rollouts per game and replay seeds are reduced, and the reduction is recorded in the results file.
5. **Frozen detector.** RefAlign-S is frozen at its default configuration (content-aware nearest reference, backbone threshold 0.9, instance numbers stripped, "Nothing happens" status compared) after inspecting development-split predictions only (no development labels exist). No parameter was tuned.
6. **Budget-driven sizes (recorded before test-split attribution).** Test collection: 8 rollouts per game (not 10). Random-reference judge variant: 2 seeds (the free deterministic variant keeps 5). Replay profile for labels: 40 reference-eligible failures, grid k = 1..10, expert and no-edit replays, 2 seeds each. Method-selected replays: a seeded subset of 100 reference-eligible failures (the 40 profiled runs plus 60 others), modes expert/repair/no-edit, 2 seeds. Label rule: strict = expert override succeeds on both seeds and the no-edit control succeeds on neither (relaxed rule reported as sensitivity).
7. **Execution log (2026-10-02).** Collection interrupted by exhausted credit at 600/1,072 episodes and resumed from saved state; replays interrupted by a disk-space leak in the simulator backend (one 35 MB library copy per environment construction) at 2,725/4,370 episodes and resumed after the adapter was changed to reuse one environment per worker process; the leave-one-type-out positional rules were fitted on the replay labels after the main batch and their selected steps replayed with the same protocol (652 episodes). Step-0 (Planner) selections admit no expert substitution and are reported separately. No saved outcome was modified by any of these events.
