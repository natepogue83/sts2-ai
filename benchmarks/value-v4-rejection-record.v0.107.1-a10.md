# Value-v4 rejection record (`v0.107.1`, Ascension 10)

## Final decision

Reject the single bounded value-v4 candidate before independent-audit
collection. Do not build v5, do not collect or open `VCAL4`, and do not execute
`SEARCHV1`. Reopening value/search requires a new explicitly approved
milestone after the base policy materially improves.

Fable reviewed this milestone after the result and returned `approve` for the
hard stop and redirect to base-policy/data quality.

## Frozen protocol

- Design precommit SHA-256:
  `ab077f8f7a5c64129ccd6493efaef40352baa9599a997eb9f57b63906cd04416`
- Fresh audit manifest SHA-256:
  `b61ffe6ff43135e8fb3819b1a39990f7dea2ad3bcb1d1a448041548a6ac3c657`
- Candidate: three-member group-balanced bootstrap nonlinear public-state
  ensemble, with explicit character interactions and explainable basis terms.
- Calibration candidates: identity, global Platt, per-character Platt.
- Selection: five-fold character-balanced seed-group out-of-fold predictions;
  internal test hidden until one calibrator was frozen.
- Attempts: one. A v5 retry was explicitly disallowed if this attempt failed.

The out-of-fold selector chose per-character Platt. Its validation ECE was
5.4882%, Brier score was 0.150335, and worst-character ECE was 12.1643%.

## One internal test

The internal test contained 1,782 states from 60 seed groups.

| Metric | Raw ensemble | Selected per-character calibration |
|---|---:|---:|
| ECE (10 bins) | **5.8789%** | 9.1161% |
| Brier score | **0.176587** | 0.180849 |
| Log loss | **0.529803** | 0.550171 |

Calibrated per-character ECE was 15.0380% Ironclad, 8.0350% Silent,
8.1561% Defect, 15.8724% Necrobinder, and 10.7686% Regent. Therefore all
three fixed criteria failed: overall ECE was worse, Brier degradation exceeded
0.002, and two characters exceeded the 15% ceiling.

The rejected checkpoint is
`checkpoints/combat_value.survive3-nonlinear-ensemble-v4.v0.107.1.npz`
(SHA-256
`339d6dc2323d041c0dab880b5c88b80b71fcc3d0ca68f40dc1bc56ef6691bb56`).
Its serialized status is `rejected_held_out_gate` and `search_eligible=false`.
The development report SHA-256 is
`321d1b519c331369410ef3f5878e3607f680d117c7b96314709fb7c0ca1e1b5c`.

## Sealed future gates

`data/raw/value-calibration-v4` contains zero files. No `VCAL4` outcome was
collected or read. No `SEARCHV1` trajectory exists, and its execution flag
remains false. These seeds are preserved rather than spent on a candidate that
already failed internally.

The next milestone is base-policy improvement: character-balanced human or
high-quality policy supervision, a source/patch compatibility gate, and a
frozen online comparison against the 0/50-win, mean-floor-8.0 Deep Sets
baseline.

