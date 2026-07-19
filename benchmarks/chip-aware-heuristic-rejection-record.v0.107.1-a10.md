# Chip-aware tactical-teacher rejection (`v0.107.1`, Ascension 10)

## Final decision

Reject `greedy_rules_v2_chip_aware` as the next DAgger teacher. Do not tune it
on `TACTV1`, and do not train the stage-2 Deep Sets candidate from it. This
honors the one-attempt stop rule in the base-policy milestone precommit.

## Frozen protocol

- Base-policy milestone:
  `benchmarks/base-policy-v2-milestone-precommit.v0.107.1-a10.json`
- Tactical manifest SHA-256:
  `a6fdfefb17bf2ba25279558ef60d3ba0bb4669652ac4dd19684bfcaa38676a7b`
- Immutable report SHA-256:
  `58b734733b208f9a43c1b1bb7e83d6cce3862b9aecd7717b6cf50643f47df3a4`
- Evaluation: 50 fresh paired seeds, exactly 10 per character and 100 total
  headless rollouts.
- Candidate-only change: increase block-card priority using preventable
  incoming damage and a lethal-damage prevention bonus.
- All noncombat rules and the Attack Potion-only safety gate were unchanged.

## Result

| Character | Baseline mean floor | Candidate mean floor | Paired delta |
|---|---:|---:|---:|
| Ironclad | 7.8 | 8.1 | +0.3 |
| Silent | 8.7 | 8.2 | -0.5 |
| Defect | 11.0 | 11.8 | +0.8 |
| Necrobinder | 8.9 | 10.4 | +1.5 |
| Regent | 9.6 | 9.1 | -0.5 |
| **Overall** | **9.20** | **9.52** | **+0.32** |

Both arms won zero runs. The candidate improved 14 pairs, tied 30, and was
worse on 6. Its descriptive paired 95% interval was -0.2685 to +0.9085 floors.

Two of four gates failed:

- overall paired mean-floor delta was +0.32, below the frozen +0.75 gate;
- candidate truncations were 1 versus baseline 0;
- both per-character -0.5 floors met the inclusive lower bound;
- victories were tied at zero.

The one truncation was Defect seed `TACTV1D000005` on floor 13. The native
runtime reported that Beam Cell remained in hand after a play action. It is
retained in the outcome as precommitted; removing or rerunning it would not
rescue the overall mean-floor gate.

## Interpretation

Chip awareness helped Necrobinder and modestly helped Defect, but a single
global block-priority adjustment was not robust across characters and made
fights materially longer. `TACTV1` is now spent evidence, not a tuning pool.
The next base-policy step must use a new precommitted approach and fresh seeds,
with character-specific tactical supervision or exact high-quality combat
actions preferred over another scalar heuristic-weight edit.
