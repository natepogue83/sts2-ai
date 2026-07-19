# Value-v3 failure analysis (`v0.107.1`, Ascension 10)

## Decision

Retire the 91-feature character-interaction linear model plus one global Platt
transform. Do not use it for search and do not make another retry in the same
family. The raw linear scores contained useful signal, but the selected global
calibrator amplified confidence in a way that did not transfer across fresh
characters, seeds, or state regions.

This memo is diagnostic. The opened v3 audit is development evidence from now
on; it is not blind evidence for any successor.

## What failed

The model failed internally before the blind audit and then failed all three
precommitted audit gates.

| Evidence | Raw ECE | Calibrated ECE | Raw Brier | Calibrated Brier |
|---|---:|---:|---:|---:|
| Internal grouped test, 1,596 states | **5.5813%** | 6.1972% | **0.166511** | 0.168678 |
| Independent audit, 1,448 states / 75 runs | **6.1952%** | 8.8058% | **0.177364** | 0.182780 |

On the independent audit, calibration improved both ECE and Brier only for
Necrobinder. It worsened both metrics for Defect, Ironclad, Regent, and Silent.

| Character | Internal test positive rate | Audit positive rate | Audit raw ECE | Audit calibrated ECE | Audit Brier change |
|---|---:|---:|---:|---:|---:|
| Ironclad | 57.92% | 57.95% | 9.2466% | 9.6161% | +0.003936 |
| Silent | 49.29% | 54.45% | 11.2884% | 14.1172% | +0.004048 |
| Defect | 63.55% | 47.60% | 12.6926% | 15.0069% | +0.010886 |
| Necrobinder | 44.60% | 59.45% | 11.8035% | **8.9732%** | **-0.004023** |
| Regent | 48.84% | 57.52% | 9.8336% | 14.2792% | +0.010327 |

The direction and size of the label-rate shift differed sharply by character:
Defect moved down 15.95 points while Necrobinder moved up 14.84 points. A
single transform could not correct both. The fitted transform had slope
`1.313919` and intercept `0.118013`, so it generally pushed high raw
probabilities higher and made the already-overconfident audit regions worse.

## Where the error concentrated

These are descriptive audit slices computed from public features only. They
are not independent causal tests because multiple states come from each run.

| Public-state slice | States | Observed | Mean raw | Mean calibrated | Brier change |
|---|---:|---:|---:|---:|---:|
| Floors 9-16 | 266 | 24.8% | 40.4% | 40.7% | +0.0185 |
| No current incoming damage | 335 | 55.2% | 64.8% | 69.3% | +0.0142 |
| Player HP above 75% | 271 | 77.5% | 81.5% | 88.2% | +0.0097 |
| Player HP 50-75% | 577 | 70.9% | 71.9% | 78.2% | +0.0042 |
| Combat round 1-2 | 658 | 55.0% | 60.4% | 64.4% | +0.0043 |

The base model's largest visible miss was later Act-1 states: it overpredicted
three-floor survival by about 15.6 points on floors 9-16, and the monotonic
global transform could not change that ranking error. The transform introduced
additional overconfidence in apparently safe states, especially high-HP and
zero-incoming-damage observations. This suggests missing nonlinear state
effects and interactions, not merely an intercept correction problem.

The raw audit ECE for every character was below the 15% ceiling. That does not
retroactively promote the raw model—the frozen rule required calibration to
improve overall ECE—but it shows why the identity transform must be an explicit
candidate in the next precommitted calibration selection rather than an
after-the-fact fallback.

## Bounded implication for v4

Use one richer, explainable, character-conditioned public-state candidate with
nonlinear state features or a small deterministic ensemble. Select exactly one
of identity, global Platt, or per-character Platt using nested seed-grouped
development predictions. Open the internal test only once after that selection.
Then freeze a new character-balanced audit manifest before collection and open
that audit once. If any unchanged promotion gate fails, stop; do not build v5.

Source artifacts:

- `value-calibration.survive3-interactions-v3.development.v0.107.1.json`
- `value-calibration.survive3-interactions-v3.audit.v0.107.1.json`
- `combat_value.survive3-interactions-v3.v0.107.1.npz`

