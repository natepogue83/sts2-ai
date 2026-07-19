# STSAI

STSAI is a research project for training an AI to play *Slay the Spire 2* at a
high level. The project deliberately separates three jobs that need different
data and learning methods:

1. **Run-history learning** — learn card rewards, pathing, shops, rest sites,
   upgrades, and long-horizon value from completed `.run` files.
2. **Combat learning** — collect full state/action trajectories through a
   user-enabled Workshop mod, then learn legal card and target choices.
3. **Search and simulation** — improve beyond demonstrations with a fast,
   patch-versioned native headless runtime and search/RL.

The current prototype includes a privacy-aware run importer, a unified macro
decision dataset, character-conditioned masked policies, a safe one-step live
agent, a fixed-seed evaluation manifest, and a process adapter for the
MIT-licensed `sts2-cli` native headless runtime. It reads local `.run` files,
public NDJSON snapshots, and gzip-compressed NDJSON without copying raw player
identifiers into the derived dataset.

## Why run histories are useful—but not sufficient

STS2 run histories record excellent high-level supervision: offered versus
picked cards, HP/gold progression, deck changes, relic choices, events, campfire
actions, encounters, turn counts, and the final outcome. They do not record the
exact card-by-card sequence and intermediate state of every combat. Therefore:

- use run histories for macro decisions and state-value pretraining;
- use Workshop telemetry for combat behavior cloning;
- use the native headless runtime for counterfactual search, self-play, and
  reinforcement learning.

Never put `win`, final floor, future relics, or other post-decision facts into a
policy input. STSAI keeps outcome fields under `label_metadata` so training code
can use them for curation or value targets without accidentally leaking them
into the observation.

## Quick start

The importer itself uses only the Python standard library.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .

# Inspect your local history (auto-detected on Windows)
stsai inspect-runs

# Build an anonymized card-reward behavior-cloning dataset
stsai build-card-dataset --output data/processed/card_choices.jsonl

# Or build the unified card, relic, potion, and Ancient-choice dataset
stsai build-decision-dataset --output data/processed/decisions.jsonl

# Install NumPy, train the masked neural baseline, and evaluate held-out seeds
pip install -e ".[train]"
stsai train-card-policy `
  --dataset data/processed/card_choices.jsonl `
  --output checkpoints/card_policy.npz
stsai evaluate-card-policy `
  --dataset data/processed/card_choices.jsonl `
  --model checkpoints/card_policy.npz
```

Despite the historical command name, the same trainer accepts the unified
decision dataset. The feature encoder includes the active character and an
explicit character-by-option interaction, so Ironclad, Silent, Defect, Regent,
and Necrobinder learn different preferences from one shared model.

Without installing the package, run it from the repository root with:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m stsai inspect-runs
```

Build the pinned native simulator against the installed v0.107.1 game files,
then verify a complete windowless process/JSON round trip:

```powershell
.\scripts\setup-sts2-cli.ps1
stsai headless-probe --character Silent --ascension 10 --seed UGL41ATF9W
```

The setup script clones a pinned upstream revision under ignored `build/`,
copies the required game assemblies there, applies the v0.107.1 compatibility
patch, and builds it. It never modifies the installed game files.

You can pass one or more sources. Directories are scanned recursively.

```powershell
stsai inspect-runs --input C:\path\to\history --input C:\path\to\snapshot.ndjson.gz

stsai build-card-dataset `
  --input C:\path\to\snapshot.ndjson.gz `
  --min-ascension 5 `
  --output data/processed/high_ascension_cards.jsonl
```

By default the builder excludes abandoned, co-op, and server-flagged cheated
runs. It groups train/validation/test splits by seed so the same deterministic
run cannot cross evaluation boundaries. Use `--help` for all filters.

## Dataset record

Each output line is one JSON decision:

```json
{
  "schema_version": 1,
  "decision_type": "card_reward",
  "split": "train",
  "run_key": "an anonymized sha256 digest",
  "patch": {"build_id": "...", "run_schema_version": 1},
  "observation": {
    "character": "CHARACTER.IRONCLAD",
    "ascension": 5,
    "act": 1,
    "floor": 3,
    "current_hp": 68,
    "max_hp": 80,
    "current_gold": 112,
    "deck": ["CARD.BASH", "CARD.DEFEND", "..."],
    "relics": ["RELIC.BURNING_BLOOD"],
    "map_point_type": "ROOM"
  },
  "legal_options": ["CARD.A", "CARD.B", "CARD.C", "SKIP"],
  "chosen": "CARD.B",
  "label_metadata": {"run_won": true, "floors_reached": 50}
}
```

The deck snapshot is reconstructed backwards from the final deck plus the
per-floor gained, removed, and transformed-card deltas. This is more robust to
balance patches than hard-coding a character's starting deck.

## Data sources

- Your own `.run` files are typically under
  `%APPDATA%\SlayTheSpire2\steam\<SteamID>\profile1\saves\history`.
- [STS2 Runs](https://sts2runs.com/downloads) publishes monthly gzip-compressed
  NDJSON snapshots and marks suspected cheated runs.
- [Spire Codex](https://spire-codex.com/) exposes patch-versioned game data and
  public run APIs. Respect its published rate limits and terms.

Do not scrape a service that does not explicitly permit bulk use. Prefer its
published snapshot, API, or a locally exported dataset, and preserve source and
license metadata alongside any raw snapshot.

## Roadmap

- [x] Local `.run`, JSON, NDJSON, and `.gz` ingestion
- [x] Filtering, anonymization, patch labels, and seed-grouped splits
- [x] Card-reward behavior-cloning records
- [x] Unified card, relic, potion, and Ancient-choice records
- [ ] Dataset builders for rest, event, shop, and route decisions
- [x] Masked neural card-reward policy baseline (51.9% held-out A10 accuracy)
- [x] Unified character-conditioned policy baseline (54.4% held-out A10 accuracy)
- [ ] State-value and other macro-decision baselines
- [x] Localhost state/action/trajectory adapter for the existing MIT bridge
- [x] Safe character-conditioned live macro recommender and one-step executor
- [x] Legal-action-masked combat trajectory dataset and combat policy baseline
- [x] Opt-in Workshop-native human combat demonstration logger
- [x] Patch/model-guarded native replay inspector and in-game replay export
- [x] Windowless official-engine oracle with managed start/stop scripts
- [x] Pinned `sts2-cli` native headless runtime for all five characters
- [x] Windowless Python process/JSON adapter and fixed-run smoke probe
- [x] Character-aware reset/step environment and legal-action masks
- [x] Atomic full-transition trajectory recorder and legal-random baseline
- [x] Character-conditioned macro/heuristic hybrid and three-way headless ablation
- [x] Character-balanced native heuristic-distillation combat corpus and checkpoint
- [x] Frozen paired online combat-distillation smoke evaluation
- [x] Independent v0.107.1 Silent replay/RNG validation kernel
- [x] Privacy-safe RunReplays audit and bounded exact-build suffix reconstruction
- [x] DAgger aggregation on learner-visited combat states
- [x] Typed Deep Sets encoder with shared legal-candidate scoring
- [x] Explainable character-aware short-horizon value candidate and calibration gate
- [ ] Search-eligible calibrated value head and policy ensemble
- [x] Process-isolated reconstructive snapshot/fork with native proof gates
- [x] Leakage-aware one-ply search plumbing without training-label emission
- [ ] Authorized search-generated policy/value labels and strength evaluation
- [ ] Native replay-to-per-action training transition reconstruction
- [ ] Batched multi-process environment manager
- [ ] Search plus offline/online reinforcement-learning loop
- [x] Character-balanced, first-attempt fixed-seed evaluation harness

The Workshop bridge will be opt-in, local-only by default, and intended for
single-player/modded research sessions. It will not bypass game protections or
operate in unmodded/competitive contexts.

## Primary native headless simulator

STSAI uses the MIT-licensed
[`wuhao21/sts2-cli`](https://github.com/wuhao21/sts2-cli) as its primary
high-throughput environment. It runs the real `sts2.dll` logic behind a JSON
stdin/stdout protocol, without opening Godot, and supports Ironclad, Silent,
Defect, Necrobinder, and Regent. This gives search and RL exact card effects,
enemy AI, relic hooks, and deterministic game RNG without requiring us to
reimplement the whole game.

The integration is pinned to both an upstream commit and STS2 v0.107.1. A game
update must fail closed until the compatibility patch and deterministic probes
are revalidated. The runtime currently initializes vanilla game models; use the
official-engine oracle below when validating Workshop-mod behavior.

The Python adapter owns one isolated process and shuts it down cleanly:

```python
from stsai.headless import Sts2HeadlessClient

with Sts2HeadlessClient() as game:
    decision = game.start_run(
        character="Silent", ascension=10, seed="UGL41ATF9W"
    )
```

### Reconstructive forks and bounded search

`Sts2HeadlessClient.snapshot()` freezes a seeded mutation transcript, the
runtime/dependency bundle fingerprint, the last native decision, and a native
semantic checkpoint. `fork()` replays that transcript in a fresh OS process
and accepts the branch only when the final decision and checkpoint match. This
avoids sharing the game's process-global run, combat, selector, localization,
and action-queue state between branches.

The protocol fails closed unless the native runtime is idle at an allowlisted
stable map or combat-play decision. Pending selections, rewards, events, rest
tasks, potion continuations, queued actions, unseeded runs, changed runtime
DLLs, and unknown mutating commands are rejected. The semantic gate covers the
run and player RNG counters, ordered combat piles, temporary card state,
creatures, monster moves and RNG, powers, relic state, orbs, action/choice IDs,
and map/run context. Wall-clock `Rng.Chaotic` is excluded because the pinned
build uses it for presentation paths rather than gameplay.

`stsai.search.OnePlyForkSearchPolicy` is the first consumer. Its initial mode is
intentionally conservative: it evaluates only immediate Strike, Defend, Bash,
and Neutralize branches using public combat state. End turn, card generation,
draw, subordinate choices, and every unproved effect fall back to the base
policy. Branch values are not written as training labels. The live five-character
plumbing and hidden-draw boundary proof are recorded in
`benchmarks/headless-one-ply-search-smoke-v0.107.1.json`.

An explicit linear value candidate now predicts the probability of surviving
the next three floors from public state. It is character-aware, seed-grouped,
and returns a plain-language summary plus its five largest feature
contributions. Training uses only completed first-party headless trajectories;
search-generated labels remain disabled. The checkpoint also carries a held-out
promotion flag, and search refuses to use a checkpoint whose flag is false.

The first candidate did not pass: overall test ECE improved from 4.82% to
4.46%, but Defect and Necrobinder ECE were 15.26% and 15.87%, above the frozen
15% per-character ceiling. The ten outcome-blind paired search cases are frozen
in `benchmarks/v0.107.1-a10-search-paired-precommit.json`, but execution remains
disabled until a value checkpoint passes every gate.

One bounded replacement added 65 explainable character-by-state interaction
terms. It also failed: calibrated test ECE was 4.77% versus 4.51% before
calibration, with Defect at 16.10%. Seed-group bootstrap intervals were wide,
so further retries on the same held-out groups were stopped.

A third and final linear-family attempt used 75 new development trajectories
and a separately frozen 75-trajectory blind audit, exactly 15 runs per
character in each pool. The checkpoint was frozen before the audit pool was
collected or read. On 1,448 independent audit states, global Platt calibration
worsened ECE from 6.20% to 8.81% and Brier score from 0.1774 to 0.1828; Defect
ECE was 15.0069%. It failed all three precommitted criteria and is permanently
marked `rejected_independent_audit_gate`. The reserved search outcomes remain
untouched. Reproduce a candidate by selecting `linear_v1` or
`character_interactions_v2`; use `--require-independent-audit` to keep a new
checkpoint locked until `audit-value-model` opens a disjoint pool:

```powershell
stsai train-value-model `
  --input data/raw/headless `
  --output checkpoints/combat_value.survive3.v0.107.1.npz `
  --metrics-output benchmarks/value-calibration.survive3.v0.107.1.json `
  --horizon-floors 3 --feature-schema linear_v1 --build-id v0.107.1
```

The frozen pool manifest and final audit are
`benchmarks/value-calibration-v3-precommit.v0.107.1-a10.json` and
`benchmarks/value-calibration.survive3-interactions-v3.audit.v0.107.1.json`.

### Training environment and trajectories

`stsai.headless_env.Sts2Env` wraps one native process in a small Gym-like API
without adding a Gymnasium dependency. `reset` fixes the character, ascension,
and run seed. Every observation carries the canonical character ID, and every
decision exposes an ordered candidate list plus a matching legal-action mask.
`step` accepts either a candidate index or the candidate itself.

Supported candidates include targeted card and potion actions, end turn, route
choices, event and rest options, card rewards, bundle/card selections, staged
chest opening and relic collection, merchant-ready actions, shop purchases and
removal, and leaving rooms. True potion-belt slots are preserved even after a
slot becomes empty. Locked, disabled, unplayable, unaffordable, sold, and
invalid-target actions are excluded. Unknown decision types fail explicitly
instead of making a guessed move.

The default reward is intentionally sparse: `+1` for victory, `-1` for defeat,
and `0` otherwise. This preserves the actual objective while the imitation
policy supplies the initial behavior. Record a reproducible legal-random
baseline episode with:

```powershell
stsai headless-rollout `
  --character Silent `
  --ascension 10 `
  --seed UGL41ATF9W `
  --policy-seed 7 `
  --max-steps 1000 `
  --output data/raw/headless/random.silent.a10.jsonl
```

The recorder writes to a temporary file during the episode and atomically
publishes the final JSONL only after a clean end or configured step cap. Each
transition includes the normalized observation, ordered candidates, mask,
chosen index/action, reward, next observation, terminal flag, and character.
Localized descriptions are excluded from training records.

### Hybrid policy and fixed-seed evaluation

The first full-run learned baseline uses `decision_policy.a10.npz` for the two
decision families it was actually trained on: the Ancient choice and card
rewards. Combat, pathing, rest sites, events, selections, and room exits use an
explicitly labeled deterministic heuristic. This is a bridge to learned combat
and search, not a claim that the current agent is fully learned.

Run one character-conditioned hybrid episode with:

```powershell
stsai headless-hybrid-rollout `
  --model checkpoints/decision_policy.a10.npz `
  --character Silent `
  --ascension 10 `
  --seed UGL41ATF9W `
  --max-steps 1000 `
  --output data/raw/headless/hybrid.silent.a10.jsonl
```

Evaluate legal-random, heuristic-only, and hybrid policies once on the first
five precommitted seeds per character:

```powershell
stsai evaluate-headless-policies `
  --manifest benchmarks/v0.107.1-a10.json `
  --model checkpoints/decision_policy.a10.npz `
  --count-per-character 5 `
  --random-policy-seed 7 `
  --max-steps 1000 `
  --build-id v0.107.1 `
  --trajectory-dir data/raw/headless/benchmark5f `
  --output benchmarks/headless-policies.a10.n5.json
```

The verified 25-seed Ascension-10 result reached mean floors 4.12 for random,
9.52 for heuristic-only, and 9.60 for hybrid, with zero truncations and zero
victories. Hybrid versus heuristic-only was 9 improved, 11 tied, and 5 worse,
for a mean difference of only +0.08 floors (descriptive paired 95% interval
-1.72 to +1.88). The learned macro choices are therefore not yet reliably
better than their ablation; the large gain over random primarily validates the
fallback and end-to-end runtime.

Every card reward carries the preceding native encounter ID, canonical monster
IDs, and turn count into the same feature schema used for training. Early card
picks are strategically useful for adding damage and block before Act-1 elites;
the model is allowed to choose `SKIP`, but no manual skip-rate threshold is
tuned on benchmark outcomes. Two explicit runtime guards are recorded in
trajectory metadata: shops currently leave because one v0.107.1 purchase path
can throw upstream, and an identical no-op event action is not repeated a third
time when another legal option exists.

### Native combat-distillation baseline

The first learned combat checkpoint is deliberately a plumbing baseline, not an
expert-play claim. No compatible human combat telemetry was present locally, so
the frozen v0.107.1 corpus uses the deterministic heuristic as its teacher on
100 fresh seeds: 20 per character, split by `(build, character, seed)` into
60 train, 20 validation, and 20 offline-test runs. The 100 trajectories contain
8,836 legal combat decisions and zero corpus truncations.

Create and collect a new precommitted corpus with:

```powershell
stsai create-headless-combat-corpus `
  --output benchmarks/headless-combat-bootstrap-v0.107.1-a10.json `
  --build-id v0.107.1 --ascension 10

stsai collect-headless-combat-corpus `
  --manifest benchmarks/headless-combat-bootstrap-v0.107.1-a10.json `
  --model checkpoints/decision_policy.a10.npz `
  --trajectory-dir data/raw/headless/combat-bootstrap-v0.107.1-a10

stsai build-headless-combat-dataset `
  --input data/raw/headless/combat-bootstrap-v0.107.1-a10 `
  --output data/processed/combat-headless-heuristic.v0.107.1.jsonl `
  --build-id v0.107.1

stsai train-combat-policy `
  --dataset data/processed/combat-headless-heuristic.v0.107.1.jsonl `
  --output checkpoints/combat_policy.heuristic.v0.107.1.npz
```

The fixed 1,024-by-128 masked MLP reached 82.99% top-1 accuracy on the offline
test split. That measures agreement with the heuristic teacher, not combat
strength. Potion accuracy was only 41.86% on 43 examples, so potion behavior is
especially under-supported.

The combat model is opt-in. In learned-combat rollouts it scores only the live
legal `combat_play` candidates, including exact card and enemy targets; every
other decision uses the heuristic-only policy. The frozen 25-case online smoke
evaluation is:

```powershell
stsai evaluate-headless-combat-distillation `
  --manifest benchmarks/v0.107.1-a10-combat-held.json `
  --macro-model checkpoints/decision_policy.a10.npz `
  --combat-model checkpoints/combat_policy.heuristic.v0.107.1.npz `
  --trajectory-dir data/raw/headless/combat-held-v0.107.1-a10 `
  --output benchmarks/headless-combat-distillation.a10.n5.json `
  --count-per-character 5 --max-steps 1000 --build-id v0.107.1
```

Combat BC averaged floor 10.32 versus 9.80 for the teacher, a paired +0.52
floors: 6 improved, 16 tied, and 3 worse, with a descriptive paired 95%
interval of -0.441 to +1.481. The learned arm had no truncations; one heuristic
Necrobinder run hit a native chained-selection timeout at floor 17 and was
retained as a truncation. Excluding that tied pair changes the mean delta only
to +0.542. Ironclad regressed from floor 9.4 to 7.8 across its five cases, so
this checkpoint is frozen but not promoted as the default combat policy.

Two DAgger rounds and a typed Deep Sets candidate scorer are now frozen. On the
same 1,740 held-out teacher decisions, DAgger round 1 reached 86.49% top-1
agreement, round 2 reached 85.11%, and the 325,409-parameter Deep Sets model
reached 90.29%. The set model explicitly pools variable-sized typed entities
and keeps character identity in the observation.

The offline gain has not yet become a reliable online strength gain. Against
DAgger round 1 on 25 fresh paired Ascension-10 cases, Deep Sets averaged +0.24
floors with a descriptive 95% interval of -1.2582 to +1.7382; neither arm won a
run. The checkpoint is therefore a stronger imitation baseline, not a high-level
player claim.

The next combat milestone is no longer another global-Platt linear retry. The
opened audit can be folded into future development, but never reused as blind
evidence. A replacement should precommit a calibration-selection procedure
(including identity/no post-calibration), use richer character-conditioned
value modeling or a small ensemble, and reserve a fresh blind pool before the
already-frozen policy-versus-search strength evaluation can run. The
reconstructive snapshot/fork boundary and public-state one-ply search harness
exist, but search-derived labels remain disabled until an evaluator passes
every gate and training authorization is explicit. A large learned world model
remains deferred because the native simulator already supplies the dynamics.

### RunReplays reconstruction fidelity

The RunReplays integration is currently an audit and reconstruction probe, not
an authorized training importer. It validates exact build, character,
ascension, trajectory prefix, snapshot/save cross-checks, runtime hash, legal
masks, and action-specific postconditions while omitting raw seeds, paths, and
commands from benchmark reports.

The frozen v0.107.1 four-character tranche reconstructed 100/100 suffix
commands for Defect, Ironclad, Necrobinder, and Regent on one Release runtime
hash. Its 400 commands include 244 masked combat decisions, staged treasure
rooms, real merchant-ready to inventory-open transitions, true-slot Foul Potion
uses, and an exact Attack Potion choose-a-card result. All four cases stopped
only at the configured 100-command cap. See
`benchmarks/runreplays-packaged-v0.107.1.json` (SHA-256
`64e32d6f02eb7e66923d26990bed785a4c39ec4500bdf9b32ebd4e2491de9061`).
The artifact is `dry_run_only` and `training_authorized=false`; replay-derived
training remains disabled pending explicit authorization and broader fidelity.

## Official-engine oracle

The installed game can run without a window using Godot's headless display
driver while retaining Steam initialization, Workshop mods, exact STS2 combat
mechanics, and the localhost bridge. This is the correctness oracle for the
standalone simulator, not the eventual high-throughput training environment.

Start or stop the managed worker from the repository root:

```powershell
.\scripts\start-sts2-headless.ps1
stsai bridge-state
.\scripts\stop-sts2-headless.ps1
```

The start script sets `SteamAppId=2868840` only in the child process, launches
STS2 with `--headless` and a hidden window style, waits for bridge health, and
stores its PID under ignored `build/headless/`. It refuses to start beside an
unmanaged or visible STS2 process. The stop script verifies both the executable
path and absence of a window before stopping that recorded PID.

On the verified v0.107.1 installation with eleven mods, the main-menu worker
used approximately 717 MiB working set. That is suitable as a golden reference
and low-volume modded trajectory generator, but too heavy for thousands of
parallel RL environments; `sts2-cli` is the primary training environment.

## Independent replay/RNG validator

`stsai.simulator` is a deliberately narrow, game-independent validation kernel,
not the primary full-game simulator. It currently
implements the v0.107.1 Silent replay's player-phase cards: Strike, Defend,
Neutralize, Survivor, Ricochet, and the unplayable Ascender's Bane. Definitions
live in `data/game/v0.107.1/silent-starter-cards.json`, and the kernel rejects a
different patch, character, unknown card, unsupported target, or unimplemented
enemy turn.

The RNG implementation matches STS2's deterministic string hashing,
SplitMix64 initialization, Xoshiro256** generator, counters, and stable/unstable
Fisher–Yates shuffles. Audit the exported replay without a game process:

```powershell
stsai sim-audit-replay `
  --replay data/raw/replays/latest.headless.v0.107.1.json `
  --catalog data/game/v0.107.1/silent-starter-cards.json

stsai sim-audit-replay --hands `
  --replay data/raw/replays/latest.headless.v0.107.1.json `
  --catalog data/game/v0.107.1/silent-starter-cards.json
```

The current native replay reaches 100% card-play catalog coverage. Its complete
five-turn hand timeline reconstructs with zero mismatches, including the extra
first-turn draw from Ring of the Snake, Ascender's Bane exhaustion, Survivor's
discard choice, and reshuffle order. This validates deck/RNG/action mechanics;
it does not yet validate enemy damage or end-turn hooks.

## Live bridge adapter

Rather than duplicate an existing mod, STSAI can consume the localhost REST API
from the MIT-licensed [STS2MCP](https://github.com/Gennadiyev/STS2MCP) project.
The adapter refuses non-localhost URLs. After the user installs and enables the
mod, inspect state without taking an action:

```powershell
stsai bridge-state
```

Explicitly act and record the complete before/action/after transition:

```powershell
stsai bridge-act `
  --action '{"action":"end_turn"}' `
  --trajectory data/raw/combat_trajectory.jsonl
```

`bridge-act` changes the active modded game. It is never called automatically
by the dataset builder or trainer. STS2MCP's published version was tested on an
older game build, so verify bridge compatibility after every STS2 patch before
collecting serious experiments.

For the installed STS2 v0.107.1 build, reproduce the verified zero-warning
bridge build with:

```powershell
.\scripts\build-sts2mcp.ps1
```

This pins STS2MCP commit `20eadeb`, applies the compatibility patch under
`integrations`, and stages the DLL, manifest, and telemetry example under
ignored `build/` output. It does not install the mod or modify the game
directory.

Once the user has installed and enabled that mod, ask the learned policy for a
non-mutating recommendation:

```powershell
stsai agent-recommend --model checkpoints/decision_policy.a10.npz
```

Execute exactly one supported macro action and record its transition only when
you intentionally want to change the active modded run:

```powershell
stsai agent-step `
  --model checkpoints/decision_policy.a10.npz `
  --trajectory data/raw/agent_trajectory.jsonl
```

The live agent canonicalizes the bridge's display name to the training
character ID and loads only that character's deck from the current save.

## Combat data and policy

Every explicit bridge action recorded by `bridge-act` or `combat-step` includes
the state before and after the action plus a stable anonymous trajectory ID.
Convert one or more append-only trajectories into combat decisions with:

```powershell
stsai build-combat-dataset `
  --input data/raw/combat_trajectory.jsonl `
  --build-id v0.107.1 `
  --output data/processed/combat_decisions.v0.107.1.jsonl

stsai train-combat-policy `
  --dataset data/processed/combat_decisions.v0.107.1.jsonl `
  --output checkpoints/combat_policy.v0.107.1.npz
```

The builder accepts only successful, state-changing actions. Its legal mask
expands single-target cards and potions over living enemies, excludes
unplayable cards, handles in-combat card-selection prompts, and groups every
trajectory into exactly one split. The normalized state preserves the active
character and character mechanics such as Regent stars, Defect orbs, and
Necrobinder pets.

After training on real trajectories, inspect or execute one combat decision:

```powershell
stsai combat-recommend `
  --model checkpoints/combat_policy.v0.107.1.npz `
  --build-id v0.107.1

stsai combat-step `
  --model checkpoints/combat_policy.v0.107.1.npz `
  --build-id v0.107.1 `
  --trajectory data/raw/combat_trajectory.jsonl
```

`combat-recommend` is read-only. `combat-step` is deliberately one action at a
time and records the result; it does not run an unattended loop. The native
heuristic-distillation checkpoint above is a source-labeled bootstrap only.
Human or search-improved transitions remain necessary before claiming expert
or high-level combat play.

### Passive human demonstrations

The patched Workshop bridge also contains an opt-in native logger for actions
made through the normal game UI. It records the local player's card plays,
targets, potion uses, and end turns while excluding actions issued through the
bridge. Telemetry is disabled by default.

After the user installs the staged mod files and launches it once, edit
`STS2_MCP.telemetry.conf` beside the installed DLL:

```json
{
  "enabled": true,
  "output": "combat_telemetry.jsonl"
}
```

Restart the game after changing the setting. The output is restricted to the
mod directory and can be passed directly to `build-combat-dataset`. Each game
session receives an anonymous trajectory ID so the complete session stays in a
single train, validation, or test split. Ally targeting is included: the live
mask and patched bridge can distinguish the player, Osty, co-op teammates, and
living enemies.

The stricter base-policy milestone uses telemetry schema v2 and a frozen
first-party corpus manifest. Schema v2 rotates its anonymous trajectory ID per
run and records the active game build, game-assembly hash, monotonic action
step, canonical character ID, and a one-way run-seed hash. It is deliberately
single-player and records no bridge-issued action or policy recommendation.

Create the manifest before collection, then build the balanced train-only
addition after at least 100 valid actions have been recorded for every
character:

```powershell
stsai create-human-combat-corpus-manifest `
  --output benchmarks/human-combat-corpus-v0.107.1-a10.precommit.json `
  --game-assembly "<game-dir>/data_sts2_windows_x86_64/sts2.dll" `
  --bridge-dll build/STS2MCP-out/STS2_MCP.dll `
  --telemetry-patch integrations/sts2mcp-human-telemetry.patch `
  --build-id v0.107.1 --ascension 10 --decisions-per-character 100

stsai build-human-combat-dataset `
  --manifest benchmarks/human-combat-corpus-v0.107.1-a10.precommit.json `
  --input "<game-dir>/mods/combat_telemetry.jsonl" `
  --output data/processed/combat-human-ui.v0.107.1.jsonl `
  --receipt benchmarks/human-combat-corpus-collection.v0.107.1-a10.json
```

The strict builder fails atomically on a wrong build or binary, non-opt-in
source, unknown character, duplicate event, reserved evaluation seed, wrong
Ascension, or action that is not uniquely legal. It samples exactly the same
number of rows per character and emits them as training data only. Passive UI
records are exact pre-action behavior-cloning demonstrations; they do not claim
an immediate post-action state or an RL transition.

## Native combat replays

STS2 writes an anonymized deterministic `latest.mcr` after combat. It contains
the initial run, ordered network actions, and an optional list of full combat
checksum checkpoints, which makes it useful for validating a simulator and
reconstructing expert combat trajectories. The format is pinned to both the
game patch and the exact model table. A replay recorded with Workshop mods
cannot be safely decoded by a base-game-only process.

Inspect only the safe header of any replay without loading its model payload:

```powershell
dotnet run --project tools/ReplayInspector -- `
  "$env:APPDATA\SlayTheSpire2\steam\<SteamID>\profile1\replays\latest.mcr"
```

Full standalone decoding requires an explicit patch guard and succeeds only
when the replay's model-table hash matches the installed base game:

```powershell
dotnet run --project tools/ReplayInspector -- `
  "$env:APPDATA\SlayTheSpire2\steam\<SteamID>\profile1\replays\latest.mcr" `
  --full --expected-version v0.107.1
```

For a replay made with Workshop mods, let the patched bridge decode it inside
the running game after those exact mods have initialized:

```powershell
stsai bridge-replay --output data/raw/replays/latest.v0.107.1.json
```

This endpoint is read-only and localhost-only. It verifies the replay model
hash against the active model table, imposes a 64 MiB input limit, and exports
player indices rather than raw player IDs. The JSON includes the active
character for every player, ordered card/potion/end-turn actions, target combat
IDs, and any checksum summaries the game recorded (piles, creatures, powers,
character mechanics, and RNG counters). The tested single-player replay
contained an action stream but zero checksum checkpoints, so checkpoints must be
treated as optional; passive telemetry is still the direct source of exact
pre-action states. STS2 keeps only `latest.mcr`, so export it after each combat
before the next completed combat overwrites it. Replay summaries are not yet
accepted directly by `build-combat-dataset`; the next step is deterministic
event replay that pairs each expert action with its exact pre-action state.
