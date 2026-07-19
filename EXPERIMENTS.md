# Experiments

## 2026-07-13 — Ascension-10 card-reward behavior cloning

### Question

Can public and local STS2 run histories provide enough supervision to learn a
meaningful card-reward policy?

### Data

- Public STS2 Runs snapshot: 6,796 raw runs.
- Local history: 404 raw runs.
- Conservative filters: solo, completed, not server-flagged as cheated,
  Ascension 10.
- Duplicates are removed by an anonymized digest of build, seed, start time, and
  character roster.
- Derived decisions: 39,039 from 2,431 runs.
- Seed-grouped split: 31,357 train / 3,958 validation / 3,724 test.
- Builds represented: v0.98.2 through v0.107.1.

Raw player ids, Steam ids, paths, and source filenames are not copied into the
derived records. Run outcome is retained only under `label_metadata` and is not
encoded as a policy feature.

### Model

- Candidate-masked one-hidden-layer MLP
- Stable feature hashing: 256 dimensions
- Hidden width: 64
- Optimizer: Adam, learning rate 0.001
- Batch size: 128 decisions
- Epochs: 12
- Seed: 7

The model scores only the choices actually offered, including `SKIP`. Deck-card
and relic-card interactions are hashed so a newly patched card remains
representable without changing the output layer.

### Results

| Split | Decisions | Exact choice | Random choice | MRR | Cross-entropy |
|---|---:|---:|---:|---:|---:|
| Train (final epoch) | 31,357 | 63.73% | — | — | 0.877 |
| Validation | 3,958 | 51.57% | 23.44% | 0.708 | 1.139 |
| Test | 3,724 | **51.88%** | **23.41%** | **0.708** | **1.131** |

The v0.107.1 held-out subset scored 58.62%, but it contains only 29 decisions
and should not be treated as a precise current-patch estimate.

### Interpretation

The run-history proposal is validated: offered-versus-picked logs contain a
strong learnable signal, and scaling from 3,092 local decisions to the public
expert-tier data improved generalization while sharply reducing train/test
overfit.

Exact-choice accuracy is not a win-rate measurement. Several cards can be
equally strong, human choices are noisy, and the dataset is not limited to elite
players. The next policy should predict a calibrated distribution/value over
choices, train with player/run quality weighting, and be evaluated by replay or
fixed-seed game outcomes.

### Reproduce

```powershell
$env:PYTHONPATH = "$PWD\src"

python -m stsai build-card-dataset `
  --input data\raw\runs-all-before-2026-06.json.gz `
  --input "$env:APPDATA\SlayTheSpire2" `
  --min-ascension 10 `
  --output data\processed\card_choices.a10.jsonl

python -m stsai train-card-policy `
  --dataset data\processed\card_choices.a10.jsonl `
  --output checkpoints\card_policy.a10.npz `
  --epochs 12 --dimension 256 --hidden 64 --batch-size 128

python -m stsai evaluate-card-policy `
  --dataset data\processed\card_choices.a10.jsonl `
  --model checkpoints\card_policy.a10.npz `
  --split test
```

## 2026-07-13 — Unified character-conditioned macro policy

The unified Ascension-10 dataset contains 54,038 decisions from completed,
solo, unflagged runs: 39,039 card rewards, 4,850 relic rewards, 5,163 potion
rewards, and 4,986 Ancient choices. Splits remain grouped by seed.

The shared masked MLP encodes both `character` and `character|option` features.
This lets data-efficient general patterns transfer between characters while
preserving character-specific card and relic preferences.

| Held-out group | Decisions | Exact choice | Random choice |
|---|---:|---:|---:|
| All decisions | 5,167 | **54.44%** | 24.65% |
| Ironclad | 881 | 50.74% | — |
| Silent | 1,087 | 57.41% | — |
| Defect | 913 | 51.81% | — |
| Regent | 1,358 | 54.20% | — |
| Necrobinder | 928 | 57.44% | — |

By decision type, exact-choice accuracy is 50.27% for cards, 65.81% for relics,
74.75% for potions, and 54.53% for Ancient choices. These are imitation
metrics, not win rates; the fixed-seed harness exists to measure actual run
performance by character and patch.

## 2026-07-13 — Headless official-engine oracle

STS2 v0.107.1 was launched directly with Godot's `--headless` driver and the
transient child-process environment `SteamAppId=2868840`. Steamworks initialized
successfully, all ten existing Workshop mods plus the patched STS2MCP bridge
loaded, and the localhost bridge returned main-menu state with no native window.

The process used approximately 717 MiB working set and 605 MiB private memory at
the main menu. It successfully decoded the existing Ascension-10 Silent native
replay using model-table hash `2309051811`. This establishes the headless game as
a fidelity oracle, but its memory footprint makes it unsuitable as the final
massively parallel simulator.

The native replay contained 25 events (15 card plays and four player end-turn
actions) but zero checksum checkpoints. Native replays therefore validate action
order and patch/model compatibility, while opt-in passive telemetry supplies the
exact pre-action observations needed for behavior cloning.

### Standalone replay reconstruction

The exported initial run contains a 14-card Silent deck, Ring of the Snake,
Ascender's Bane, run seed `UGL41ATF9W`, and zero initial Shuffle/CombatTargets
counters. The standalone kernel reproduces STS2's deterministic hash and
Xoshiro256** streams. An independent .NET probe and Python returned the same
first eight two-target samples: `1,0,0,1,1,0,1,1`.

The patch-derived catalog implements every card played in the replay, giving
15/15 card-play coverage. Initial unstable shuffle, Ring of the Snake's seven
card draw, Survivor's recorded combat-card discard, Ascender's Bane exhaustion,
and stable reshuffles reconstruct all five turn-start hands with zero action or
choice mismatches. Enemy turns and hook-driven damage remain outside this first
kernel and fail explicitly rather than being approximated.

## 2026-07-13 — Native `sts2-cli` integration

The project pivoted from implementing the remaining full game rules locally to
using the MIT-licensed `wuhao21/sts2-cli` runtime. It executes the real game
assembly without Godot and exposes full-run decisions and actions through a
line-delimited JSON protocol. All five playable characters are supported.

Upstream commit `d11aa883b582dd68bd39b331f3370746b30d447e` required three
isolated compatibility changes for the installed STS2 v0.107.1 build: the
renamed async saved-run setup method, .NET 9 roll-forward, and explicit skipped
mod initialization. The setup process patches only a copied `sts2.dll` under
ignored `build/`; the Steam installation is not modified.

A real-process smoke test started an Ascension-10 Silent run with seed
`UGL41ATF9W` and returned the deterministic Neow decision with Fishing Rod,
Lost Coffer, and Hefty Tablet. Protocol startup, fixed-seed run creation, JSON
decoding, and graceful shutdown completed in under one second after build. The
official Godot worker remains the golden oracle for Workshop-mod behavior; the
small independent Python kernel remains useful for catching RNG and replay
regressions.

### Character-aware environment baseline

The native runtime is now wrapped by a dependency-free reset/step environment.
It converts every supported decision into an ordered dynamic candidate set and
legal mask, keeps the canonical character in every observation, applies sparse
terminal rewards, and atomically records full transitions. Unit fixtures cover
target expansion, unplayable cards, locked event choices, multi-card
combinations, shop affordability, terminal handling, rewards, and recorder
publication.

The first complete legal-random Ascension-10 Silent episode used game seed
`UGL41ATF9W` and policy seed `7`. It made 30 legal decisions and lost on Act 1,
floor 2 for reward `-1`. This is a plumbing baseline, not a competitive policy;
the next learning milestone is replacing uniform candidates with the existing
character-conditioned imitation scores and measuring fixed-seed improvement.

A character-balanced pass then used game seed `STSAI_PROBE` and policy seed `7`:

| Character | Steps | Final act | Final floor | Reward |
|---|---:|---:|---:|---:|
| Ironclad | 60 | 1 | 5 | -1 |
| Silent | 39 | 1 | 4 | -1 |
| Defect | 68 | 1 | 6 | -1 |
| Necrobinder | 42 | 1 | 4 | -1 |
| Regent | 63 | 1 | 6 | -1 |

All five episodes reached a native `game_over` decision without an illegal
candidate, runtime error, step cap, visible window, or partially published
trajectory.

## 2026-07-13 — Character-conditioned hybrid headless benchmark

### Question

Does the existing character-conditioned macro imitation checkpoint improve a
complete native headless agent once unsupported decisions use the same explicit
heuristic fallback?

### Protocol

- Runtime: pinned `sts2-cli` integration against STS2 v0.107.1.
- Difficulty: Ascension 10.
- Cases: the first five precommitted seeds for each of the five characters in
  `benchmarks/v0.107.1-a10.json`; selection never inspects policy outcomes.
- Policies: legal-random with a deterministic per-case policy seed,
  heuristic-only, and hybrid.
- Hybrid learning boundary: the unified macro checkpoint scores Ancient and
  card-reward choices; every other decision uses the identical heuristic-only
  fallback.
- Objective: sparse terminal reward and final floor; each arm has a 1,000-step
  cap and atomically publishes its full transition trajectory.

The checkpoint is
`476701d84a7f6cf85b1aa6a910877334c13f52ea637af967a8f6da400ca878f9`
(SHA-256). The final summary is `benchmarks/headless-policies.a10.n5.json`, and
the 75 trajectories are under ignored `data/raw/headless/benchmark5f/`.

### Live feature-schema correction

The first adapter omitted the exact preceding encounter context at card
rewards. Because the feature hasher still encoded a missing `room=None` value,
arbitrary collisions and missing encounter identity caused 85 of 88 live card
rewards (96.6%) to predict `SKIP`. That version averaged floor 8.12, below the
same heuristic fallback at 9.40, so it was rejected rather than tuned.

The native protocol now exports encounter IDs, canonical monster IDs, and turn
counts. The environment carries that combat context into the following reward,
and the policy builds the same fields used by the offline dataset. All 106 card
rewards in the final hybrid benchmark have complete context. The model selected
a card 104 times and skipped twice.

Low skipping is directionally sensible in Act 1 because early damage and block
reduce chip damage and stabilize later fights. In the held-out human test data,
the actual skip rate is 24.8% for Act-1 card rewards versus 35.8% across all
acts; the checkpoint predicts `SKIP` on 19.3% and 33.6% of those groups,
respectively. The live 1.9% rate remains more aggressive than the offline Act-1
distribution, but no prior or threshold was tuned on the held benchmark.

### Runtime guards

The v0.107.1 simulator's `Petrified Toad` shop purchase path produced an
upstream null-reference exception, so all compared policies leave shops and
record that guard. `Whispering Hollow` also keeps its already-used gold option
legal after it becomes a no-op. The deterministic policies now detect an
identical event observation plus the same proposed action, choose a different
legal option once, and record the override. The two exact capped reproductions
then terminated naturally at steps 130 and 150. The final benchmark has 75
published trajectories, no temporary files, and zero truncations.

### Results

| Character | Cases | Random mean floor | Heuristic mean floor | Hybrid mean floor | Hybrid − heuristic |
|---|---:|---:|---:|---:|---:|
| Defect | 5 | 5.2 | 9.0 | 10.2 | +1.2 |
| Ironclad | 5 | 4.6 | 12.2 | 12.8 | +0.6 |
| Necrobinder | 5 | 3.2 | 7.8 | 6.4 | -1.4 |
| Regent | 5 | 3.2 | 9.4 | 9.6 | +0.2 |
| Silent | 5 | 4.4 | 9.2 | 9.0 | -0.2 |
| **Overall** | **25** | **4.12** | **9.52** | **9.60** | **+0.08** |

All 75 arms ended in Act 1 with zero victories. Relative to random, hybrid was
better on 23 seeds, tied on one, and worse on one, for +5.48 mean floors. The
descriptive paired 95% t interval is +3.84 to +7.12. Relative to heuristic-only,
hybrid was better on 9 seeds, tied on 11, and worse on 5; its +0.08 mean-floor
difference has a paired 95% interval of -1.72 to +1.88.

### Interpretation

The full native pipeline, dynamic legal masks, all-character conditioning,
macro feature adapter, and deterministic evaluation harness work end to end.
The learned macro component recovered from the schema failure, but this sample
does not establish a reliable advantage over heuristic-only behavior. The
character rows contain only five cases each and must not be read as stable
rankings.

The next empirical milestone is a learned combat policy trained on real
per-action demonstrations, followed by a larger precommitted evaluation after
the policy is frozen. Route, shop, rest, and general event learning should then
replace additional heuristic branches before search or reinforcement learning
is credited for high-level play.

### Reproduce

```powershell
$env:PYTHONPATH = "$PWD\src"

python -m stsai evaluate-headless-policies `
  --manifest benchmarks/v0.107.1-a10.json `
  --model checkpoints/decision_policy.a10.npz `
  --output benchmarks/headless-policies.a10.n5.json `
  --trajectory-dir data/raw/headless/benchmark5f `
  --count-per-character 5 `
  --random-policy-seed 7 `
  --max-steps 1000 `
  --build-id v0.107.1
```

## 2026-07-13 — Native combat heuristic distillation

### Question

Can a character-conditioned policy learn the native heuristic's combat choices
and execute them legally in complete headless runs, before human demonstrations
or search-improved labels are available?

### Protocol

- Runtime: pinned `sts2-cli` integration against STS2 v0.107.1.
- Difficulty: Ascension 10, all five characters.
- Corpus: 100 precommitted fresh seeds, 20 per character, with exact
  `(build, character, seed)` grouping into 12 train, 4 validation, and 4 test
  runs per character.
- Teacher: deterministic heuristic-only policy.
- Dataset: 8,836 legal combat decisions from 100 complete trajectories with
  zero truncations.
- Model: 1,024 hashed features, hidden width 128, character included in every
  state, and a shared score over the current legal candidate set.
- Frozen online evaluation: five unseen seeds per character. The learned arm
  uses the combat checkpoint only at `combat_play`; both arms use the identical
  heuristic for every non-combat decision.

The dataset SHA-256 is
`746feac9529f20e33de18eec183709ffe93358c6531f4e20201aa112a2f88b3`.
The checkpoint SHA-256 is
`34e522e653130c6af724f8de1ac72a1559eb13d07ef700d93bbc009d0b46b9d2`.

### Offline results

| Metric | Result |
|---|---:|
| Test decisions | 1,740 |
| Top-1 teacher agreement | 82.99% |
| Random-candidate baseline | 34.14% |
| Mean reciprocal rank | 0.903 |
| End-turn top-1 | 97.92% |
| Play-card top-1 | 79.97% |
| Use-potion top-1 | 41.86% (43 examples) |

The frozen offline result is
`benchmarks/combat-policy-offline.heuristic.v0.107.1.json`. These values measure
teacher imitation, not combat strength.

### Frozen online results

| Character | Cases | Heuristic floor | Combat BC floor | Delta |
|---|---:|---:|---:|---:|
| Defect | 5 | 8.0 | 9.4 | +1.4 |
| Ironclad | 5 | 9.4 | 7.8 | -1.6 |
| Necrobinder | 5 | 8.8 | 9.2 | +0.4 |
| Regent | 5 | 10.2 | 10.8 | +0.6 |
| Silent | 5 | 12.6 | 14.4 | +1.8 |
| **Overall** | **25** | **9.80** | **10.32** | **+0.52** |

Across paired cases, combat BC was better on 6, tied on 16, and worse on 3.
The descriptive paired 95% normal interval was -0.441 to +1.481 floors. Neither
arm won a run. The learned arm had zero truncations. One heuristic Necrobinder
case hit a native chained-selection timeout at floor 17; it was retained and
counted as a runtime failure rather than removed. Excluding that tied pair gives
a +0.542-floor mean delta across the remaining 24 cases.

### Interpretation

The result validates character-aware state conversion, dynamic legal action
scoring, checkpoint integration, and paired native evaluation. It does not
justify promotion: the interval is inconclusive, Ironclad regressed on three of
five cases, potion supervision is sparse, and the teacher is not expert.

The checkpoint is therefore frozen as a reproducible baseline. The next
milestone is learner-state aggregation (DAgger) to expose covariate shift,
followed by a small typed set encoder and shared candidate scorer. Exact native
snapshot/fork support can later provide search-improved policy and value labels;
a learned world model is unnecessary while the native simulator supplies the
dynamics.

### Reproduce

```powershell
$env:PYTHONPATH = "$PWD\src"

python -m stsai evaluate-headless-combat-distillation `
  --manifest benchmarks/v0.107.1-a10-combat-held.json `
  --macro-model checkpoints/decision_policy.a10.npz `
  --combat-model checkpoints/combat_policy.heuristic.v0.107.1.npz `
  --trajectory-dir data/raw/headless/combat-held-v0.107.1-a10 `
  --output benchmarks/headless-combat-distillation.a10.n5.json `
  --count-per-character 5 `
  --max-steps 1000 `
  --command-timeout 15 `
  --build-id v0.107.1
```

## 2026-07-14 - DAgger and typed set-policy checkpoint

### Question

Do learner-state aggregation and a permutation-aware typed set encoder improve
the character-conditioned combat imitation baseline without overstating online
strength?

### Frozen evidence

All offline comparisons use the same 1,740 held-out teacher decisions:

| Model | Held-out top-1 | MRR | Potion top-1 |
|---|---:|---:|---:|
| Heuristic-distillation MLP | 82.99% | 0.903 | 41.86% |
| DAgger round 1 MLP | 86.49% | 0.925 | 27.91% |
| DAgger round 2 MLP | 85.11% | 0.918 | 41.86% |
| Typed Deep Sets round 2 | **90.29%** | **0.945** | **88.37%** |

The typed model is a 325,409-parameter shared legal-candidate scorer. It pools
variable-sized typed state entities and keeps the canonical character explicit.
The frozen checkpoint is
`checkpoints/combat_policy.deepsets-r2.v0.107.1.pt` (SHA-256
`5bbb35de90351827374c9188351cc11c5ce867053c057e1a45978a60bb7633e1`).

Fresh paired Ascension-10 online evaluations remain inconclusive:

- DAgger round 1 versus the heuristic-distillation MLP: +1.24 mean floors,
  descriptive 95% interval -0.1333 to +2.6133.
- DAgger round 2 versus round 1: -0.04 mean floors, interval -0.4398 to
  +0.3598.
- Deep Sets versus DAgger round 1: +0.24 mean floors, interval -1.2582 to
  +1.7382.

Each comparison contains 25 character-balanced cases; no evaluated arm won a
run. Deep Sets is promoted only as the strongest teacher-agreement checkpoint,
not as a proven stronger game-playing policy.

### Research-guided next strategy

The next work should reduce backtracking by following a staged evidence ladder:

1. Add permutation/equivariance tests for every set-valued input and compare
   sum, mean, max, count, or small attention pooling.
2. Ablate learned character embeddings and FiLM-style conditioning with
   character-balanced batches and per-character metrics.
3. Add a state-value head and train policy/value jointly on exact native
   outcomes; calibrate probabilities and compare small ensembles before search.
4. Implement exact native snapshot/fork, then use shallow search and expert
   iteration for improved policy/value targets instead of learning a world
   model the simulator already provides.
5. Consider offline IQL/CQL only after replay transitions are fidelity-complete,
   provenance-safe, and explicitly authorized for training.

Primary method references: [Deep Sets](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html),
[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html),
[FiLM](https://ojs.aaai.org/index.php/AAAI/article/view/11671),
[DAgger](https://proceedings.mlr.press/v15/ross11a.html),
[Expert Iteration](https://proceedings.neurips.cc/paper_files/paper/2017/hash/d8e1344e27a5b08cdfd5d027d9b8d6de-Abstract.html),
[Gumbel AlphaZero](https://openreview.net/pdf?id=bERaNdoegnO),
[Implicit Q-Learning](https://openreview.net/pdf?id=68n2s9ZJWF8),
[Conservative Q-Learning](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html),
[deep ensembles](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html),
and [modern calibration](https://proceedings.mlr.press/v70/guo17a.html).

## 2026-07-14 - Reconstructive forks and leakage-aware one-ply search

### Architecture decision

Branches run in fresh OS processes. The pinned engine has process-global run,
combat, card-selector, localization, reward, and action-queue state, so cloning
or running multiple simulators inside one process would not provide defensible
isolation. A snapshot is therefore an immutable seeded transcript plus the
runtime bundle hash, final native response, and native semantic checkpoint.

Protocol `0.3.0` adds `snapshot_checkpoint`. Capture is initially allowlisted
only at quiescent map and combat-play decisions. It rejects pending selectors,
rewards, event/rest/potion continuations, synchronization callbacks, running or
queued actions, unseeded transcripts, unknown mutations, schema mismatches, and
runtime DLL changes. The checkpoint includes all run/player RNG counter vectors
and critical map, player, pile, temporary card, orb, creature, monster-move,
power, relic, and action/choice state. `Rng.Chaotic` is explicitly excluded as
presentation-only in the pinned build.

This is reconstructive forking, not an in-memory engine clone, and the support
claim is limited to the allowlisted stable states and v0.107.1 runtime bundle.
Transient asynchronous Neow responses may differ while a prefix is replayed;
the fork is accepted only if its final response and semantic checkpoint converge.

### Verification

The Release runtime bundle SHA-256 is
`792748a35e701183369751018ecd546914c1742f8d6131f8d25ede1ef18dc88a`.
The expanded upstream integration patch SHA-256 is
`e0a2a35adec249498f612f69ec1dd8f926a9189ab8d92cdbb6e3ca3d51499449`;
it applies cleanly to pinned commit `d11aa883b582dd68bd39b331f3370746b30d447e`
and reproduces all 12 patched files after newline normalization.
Twenty consecutive forks of one frozen combat root produced 20 distinct child
PIDs, 20 successful exact final checkpoints, zero transient mismatches, and no
change to the parent. Native tests also prove that event and Attack Potion
selection states fail closed.

The focused native snapshot suite passes 3/3, and both Debug and Release builds
complete with zero warnings or errors. A broad native pytest run is not claimed
as passing: its first pre-existing `test_card_reward_after_combat` auto-combat
case kept the same idle .NET worker for more than six minutes. A verbose rerun
reproduced the same test-level stall and the verified pytest process tree was
stopped. The test does not call `snapshot_checkpoint`; this remains a logged
suite-health issue rather than hidden evidence for this milestone.

The first search wrapper is sequential, depth one, and
`public_deterministic_only`. It searches only immediate Strike, Defend, Bash,
and Neutralize effects. End turn is excluded because it reveals a hidden draw
and consumes enemy RNG; generated cards and subordinate selections are also
excluded. The leaf evaluator is an untrained public-state heuristic used to
validate plumbing, and the search metadata explicitly records
`training_labels_emitted=false`.

All five characters reached a first combat in the live smoke. Ironclad, Silent,
and Defect evaluated 11, 14, and 3 child processes respectively. Necrobinder
and Regent fell back because their base choices were not allowlisted. Search
kept the base action in all five cases. In a separate leakage test, two Ironclad
roots had byte-identical public decisions but different hidden draw-pile orders
and different semantic checkpoint hashes; four-node search selected the same
Bash action for both roots.

These results establish isolation and the information boundary, not playing
strength. The compact report is
`benchmarks/headless-one-ply-search-smoke-v0.107.1.json`. No external replay or
search-derived training data was used. The next evidence step is a calibrated
state-value head and a precommitted paired policy-alone versus policy-plus-search
outcome smoke before considering expert-iteration labels.

### Character-aware value candidate and promotion gate

The next interface is implemented as an explainable linear public-state value.
Its target is deliberately narrower than win probability: the chance of
surviving the next three floors under the logged-policy distribution. Features
explicitly include the active character plus public health, block, resources,
combat round, hand/playability, enemy health/block/intents, powers, orbs, stars,
and pet health. The live explanation reports a summary and the five largest
log-odds contributions. Seeds and hidden native state are not features.

The corpus uses only completed first-party native trajectories already produced
by this project. After excluding 13 invalid/incomplete files and deduplicating
4,823 repeated seed/state observations, it contains 13,105 first states from
each combat round across 880 trajectories. Outcome-blind SHA-256 ranking within each
character assigns 344 seed groups to train, 42 to calibration, and 42 to test;
the corresponding state counts are 10,671 / 1,059 / 1,375. Search labels and
external replay data were not used.

Monotonic Platt calibration is fitted on the validation split with equal total
weight per seed group. On held-out test states:

| Metric | Raw linear value | Calibrated value |
|---|---:|---:|
| ECE (10 bins) | 4.815% | **4.459%** |
| Brier score | **0.169240** | 0.170696 |
| Log loss | **0.511499** | 0.511582 |
| Accuracy at 0.5 | 73.96% | 74.18% |

Per-character calibrated ECE is 11.04% Ironclad, 14.19% Silent, 15.26% Defect,
15.87% Necrobinder, and 13.02% Regent. The frozen promotion rule required
overall ECE improvement, no more than 0.002 Brier degradation, and at most 15%
ECE for every character. The first two conditions pass; the per-character gate
fails. The checkpoint is therefore marked `rejected_held_out_gate` and
`search_eligible=false`, and `OnePlyForkSearchPolicy` refuses it.

The rejected checkpoint is
`checkpoints/combat_value.survive3.v0.107.1.npz` (SHA-256
`91876700d1410e28da04c763830dfb98f4c7c8c060dcef38a2a08dd674060744`).
The full calibration report is
`benchmarks/value-calibration.survive3.v0.107.1.json` (SHA-256
`88d3a45981898c0d204ade1032015ce6c64f73c4b30c572ede98aee017ee51a1`).
Five live character roots returned matching character metadata and factor-level
explanations, while all five were correctly refused as search inputs.

Ten fresh character-balanced outcome seeds are frozen in
`benchmarks/v0.107.1-a10-search-paired-precommit.json`. Their outcomes have not
been run or inspected, and execution remains disabled until a replacement
value checkpoint passes the same gate. This is a negative but useful result: it
prevents an unevenly calibrated evaluator from silently steering search.

One bounded replacement kept the identical target, split, and gate while adding
65 explicit character-by-state interaction terms (91 total features). It also
failed. Calibrated test ECE was 4.767% versus 4.512% for its raw scores, and
Defect remained at 16.10%; Regent was 15.06%. The SHA-256 values are
`9184fb97c77763661ed56ea3ccdd98de912ecd8c3cf829c07dd15b5d4d41f5f9`
for `checkpoints/combat_value.survive3-interactions-v2.v0.107.1.npz` and
`14c89df1bfeb3079ff00ba8f3d88d7741e2fcd78d5646081ee9d4bcde130fe84`
for its calibration report.

Seed-group bootstrap 95% ECE intervals for the replacement were wide:
Ironclad 8.02-20.71%, Silent 7.19-30.42%, Defect 11.71-26.10%, Necrobinder
11.95-29.32%, and Regent 10.74-24.67%. This confirms that another architecture
retry on the same held-out groups would risk evaluation overfitting. The next
input should be additional independent first-party calibration groups, not the
reserved search-outcome seeds.

### Independent value calibration tranche

A new manifest froze development and audit seeds before either pool was run:
`benchmarks/value-calibration-v3-precommit.v0.107.1-a10.json` (SHA-256
`2794d9e4d3359f4caa968c484f38e5c7848a2fa4626c8af10e89774498a28af7`).
Each pool contains 15 Ascension-10 runs for each of the five characters. The
ten `SEARCHV1` policy-versus-search seeds were explicitly excluded. Collection
used the first-party native runtime, the fixed Deep Sets combat policy, and the
same deterministic noncombat policy. Both receipts report 75/75 completed and
`outcomes_reported=false`; their corpus hashes are
`dc2311ca2cb6d7d629f3f803e22c9de7dd167be7a7c91fbc872a3efb68a221a0`
for development and
`ef9a7ad7bb9b7bd88a592bc5f3593105cdb2a019046ed9294af29e9f6bd05d8e`
for audit.

Collection exposed four deterministic runtime boundaries. The final adapter
permits only the proven Attack Potion subordinate selection, skips only The
Future of Potions with zero owned potions, accepts a finished-event option
postlude only after every native reward was proven selected, and avoids Rest
with Tiny Mailbox when its two potion rewards would overflow the belt. Every
guard is narrow and recorded. Focused regressions cover the empty and
potion-owned event paths, the exact Neow reward postlude, and the Tiny Mailbox
policy choice. Debug and Release builds completed with zero warnings/errors;
the current Release runtime bundle SHA-256 is
`b8e70e8faaec989d227d7763466521eaa6b7ed382e4f550abbae3ded1b15b80a`,
and the three focused snapshot tests pass. A broader native event/rest pytest run
again stalled after its first two tests and is not claimed as passing.

One replacement was trained after development collection and before any audit
trajectory existed. It kept the three-floor target, 91-feature explicit
character-interaction schema, and global monotonic Platt calibration. The
expanded development corpus contains 14,641 states across 955 trajectories and
503 seed groups. Its internal test already failed all criteria: calibrated ECE
was 6.197% versus 5.581% raw, Brier was 0.168678 versus 0.166511 raw, and
Necrobinder ECE was 15.892%. Regardless of that result, the serialized model
remained `awaiting_independent_audit` and `search_eligible=false`. Its frozen
pre-audit SHA-256 was
`6a9a7ae0e33d354c096115456216fa67d967179d3273c982e6bf291f7f9b9f79`.

The audit pool was then collected under the frozen model and opened exactly
once. It contains 1,448 first-combat-round states across all 75 audit groups:

| Metric | Raw linear probability | Global Platt calibrated |
|---|---:|---:|
| ECE (10 bins) | **6.195%** | 8.806% |
| Brier score | **0.177364** | 0.182780 |
| Log loss | **0.531571** | 0.544544 |

Calibrated per-character ECE was 9.616% Ironclad, 14.117% Silent,
15.0069% Defect, 8.973% Necrobinder, and 14.279% Regent. Therefore overall ECE
did not improve, Brier degradation exceeded 0.002, and the strict
per-character ceiling also failed. The model is permanently
`rejected_independent_audit_gate` and remains ineligible for search. The final
checkpoint SHA-256 is
`cc2410fc52a27ae3563d563ccfcacf91cf15e0740ec523108a6e5b347e00873f`;
the immutable audit report SHA-256 is
`97c792859999d5c4717706334fe5fdc3f6986b7d22dba05d8368d046d4604748`.

This result stops further retries of the same global-Platt linear family. The
opened audit may become future development data but cannot be reused as blind
evidence. A future attempt should precommit calibration-model selection
(including the identity transform), introduce richer character-conditioned
value capacity or a small ensemble, and freeze a new independent audit pool.
The reserved policy-versus-search outcomes remain unexecuted.

The follow-up diagnostic is frozen in
`benchmarks/value-v3-failure-analysis.v0.107.1.md`. It identifies a
character-dependent label shift, a global calibration transform that improved
only Necrobinder on audit, and large descriptive misses on floors 9-16,
high-health states, and zero-current-incoming-damage states. These findings set
the bounded v4 family; the opened audit is not reusable as blind evidence.

### Fresh typed-policy online baseline

Before value-v4 work, a measurement-only manifest bound the current Deep Sets
combat checkpoint and DAgger-r2 macro checkpoint to 50 fresh `BASEV1` seeds,
exactly 10 per character. It explicitly excludes all 10 `SEARCHV1` seeds, all
150 value-v3 development/audit seeds, and the prior 25-seed Deep Sets
comparison manifest. Only the proven Attack Potion path was enabled.

The one-shot run completed all 50 cases without a truncation or runtime error.
It recorded 0 victories, overall mean floor 8.0, and median floor 7. Per-character
mean floors were 8.9 Ironclad, 8.5 Silent, 7.8 Defect, 6.7 Necrobinder, and 8.1
Regent; Silent reached the maximum floor of 17. This is the requested cheap
online baseline, not a promotion test or tuning set. The frozen manifest and
report are `benchmarks/headless-combat-deepsets-online-baseline-v0.107.1-a10.precommit.json`
and `benchmarks/headless-combat-deepsets-online-baseline-v0.107.1-a10.n10.json`.

### Bounded value-v4 hard stop

One final value attempt was fully precommitted in
`benchmarks/value-v4-design-precommit.v0.107.1-a10.json`. It used a
three-member group-balanced nonlinear public-state ensemble and selected among
identity, global Platt, and per-character Platt with character-balanced
seed-group out-of-fold predictions. A fresh 75-run `VCAL4` audit was frozen
before training.

Per-character Platt won the nested development selection, but the single
internal test failed all three promotion criteria. Across 1,782 states from 60
groups, raw versus calibrated ECE was 5.8789% versus 9.1161%, and Brier was
0.176587 versus 0.180849. Ironclad calibrated ECE was 15.0380% and Necrobinder
was 15.8724%, both over the 15% ceiling. The checkpoint is
`rejected_held_out_gate` and is not search eligible.

The no-v5 stop rule is now active. `VCAL4` was not collected or opened, and
`SEARCHV1` remains unexecuted. Fable approved closing this line and redirecting
to base-policy/data improvement. The complete immutable decision is in
`benchmarks/value-v4-rejection-record.v0.107.1-a10.md`.

### First base-policy tactical screen

The base-policy milestone was frozen in
`benchmarks/base-policy-v2-milestone-precommit.v0.107.1-a10.json`. Its first
stage tested exactly one change before outcomes: a chip-aware tactical teacher
that increases block-card priority using preventable incoming damage. The
`TACTV1` manifest reserved 50 fresh seeds, 10 per character, and paired both
teacher versions on every seed with all noncombat behavior unchanged.

The candidate failed the frozen promotion gate. It raised overall mean floor
from 9.20 to 9.52, a paired gain of only +0.32 versus the required +0.75.
Per-character deltas were +0.3 Ironclad, -0.5 Silent, +0.8 Defect, +1.5
Necrobinder, and -0.5 Regent. Both arms won zero runs. One candidate Defect
trajectory truncated on a native Beam Cell play verification error, versus
zero baseline truncations. The failure is retained, although the mean-floor
gate would still fail without it.

`greedy_rules_v2_chip_aware` is rejected as the next DAgger teacher. It will
not be tuned on `TACTV1`, and the contingent Deep Sets training stage will not
run. The immutable decision is in
`benchmarks/chip-aware-heuristic-rejection-record.v0.107.1-a10.md`.

Fable reviewed the failed tactical screen and approved the data-first pivot:
collect a bounded, opt-in, first-party five-character combat corpus, train one
candidate, and return to a fresh paired online gate. Public run histories stay
macro-only; value-v5, `VCAL4`, and `SEARCHV1` remain sealed.

The next corpus is precommitted in
`benchmarks/human-combat-corpus-v0.107.1-a10.precommit.json`. It requires exactly
100 accepted Ascension-10 UI actions per character (500 total), with no policy
suggestions during collection. The schema-v2 Workshop collector and strict
Python ingestor now bind every event to the real v0.107.1 release and game
assembly, rotate anonymous trajectory IDs per run, preserve canonical character
IDs and character mechanics, hash-check all reserved evaluation seeds, reject
duplicate or non-uniquely-legal actions, and emit balanced train-only rows.

The staged bridge SHA-256 is
`e22136744a8d6f42fac1b21a827ec28a817867715cd76ea7741520281c547fcc`;
the active game assembly SHA-256 is
`a1f9e653f1e28e4076558fee1e60d218619cb7e057b887c6417f62c62c6d7a52`;
the frozen corpus manifest SHA-256 is
`cd49a167bc2d12c361a2de1510c68327f672f109b0e272dab07d4d299e82f8da`.
The bridge rebuild succeeded with zero warnings and errors, a clean pinned
upstream patch replay succeeded, and 33 focused Python tests pass. No human
telemetry exists yet: the installed bridge is still the prior binary and its
opt-in telemetry config remains disabled.

## 2026-07-14 - RunReplays fidelity tranche

### Scope and privacy

The source is the pinned RunReplays revision
`b0d2302ee69bf2ad735e0b6b51aea02408e9ef62`, reconstructed against exact game
build `v0.107.1`. Reports persist hashes and typed aggregate evidence, not raw
replay commands, local source paths, inline comments, or seeds. The accidental
case-mismatched Silent artifact remains explicitly excluded and is not deleted
or relabeled.

This tranche is an audit only. Every report and aggregate sets
`dry_run_only=true` and `training_authorized=false`.

### Fidelity corrections

The runtime and mapper now preserve native decision order and verify effects:

- event reward sets, Fake Merchant inventory, rest-site choices, purchases,
  card removal, potion discards, and room-to-map transitions;
- treasure `unopened -> relic_ready -> proceed`, with normal rewards before
  extra rewards, exact synchronizer result/player, session closure, and one
  inventory grant;
- an exact state-and-proof-gated no-op for RunReplays' duplicate
  `NetPickRelicAction` echo after the verified chest take;
- normal merchant `merchant_ready -> inventory_open`, with purchases hidden
  until the real `OpenShop` transition;
- true potion-belt slot indices, null targeting for `TargetedNoCreature`, no
  silent fallback discard, exact Foul Potion consumption/history/+100-gold
  proof, and exact Attack Potion selected-object-in-hand proof.

Direct protocol tests covered two sequential treasure sessions, duplicate and
wrong-stage rejection, three Foul Potions in slots 0/1/2, a single shop-open
transition, and an Attack Potion choose-a-card selection. The final Release
build completed with zero warnings and zero errors; 105 root unit tests passed.

### Frozen four-character result

All four cases used the packaged headless runtime SHA-256
`a618d72cd90361cab4984330c08ce030ef062b18bef25e10216974812a0cec05`.
The reconstruction-tranche integration patch SHA-256 at that time was
`ccf8f1f17e6a4834ecf7b5d66873eff8c995a04109855e8c1f4e59dd0a3928d2`.

| Character | Commands | Combat decisions | Stop |
|---|---:|---:|---|
| Defect | 100/100 | 54 | configured cap |
| Ironclad | 100/100 | 66 | configured cap |
| Necrobinder | 100/100 | 59 | configured cap |
| Regent | 100/100 | 65 | configured cap |
| **Total** | **400/400** | **244** | **four configured caps** |

Every combat decision has a complete legal mask with the chosen action legal,
and every build/trajectory/save validation gate passed. The aggregate is
`benchmarks/runreplays-packaged-v0.107.1.json`, SHA-256
`64e32d6f02eb7e66923d26990bed785a4c39ec4500bdf9b32ebd4e2491de9061`.
The four immutable case hashes are embedded in that aggregate.

### Interpretation

This proves bounded exact reconstruction for four 100-command suffixes. It does
not prove complete RunReplays coverage, full-run reconstruction, data license
authorization, or policy strength. The next replay step is broader
fidelity/provenance coverage and explicit authorization before any replay data
is converted into training transitions.
