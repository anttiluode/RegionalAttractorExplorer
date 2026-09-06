# Pilot-field audit 2026

The old **conductor** story is retired as a biological claim. The replacement is a narrower, falsifiable geometry question:

> **Does the spatial phase field of a slow oscillation contain predictive information about where faster activity moves next?**

And, separately:

> **Does the current fast-activity pattern contain predictive information about how the slow phase field changes next?**

The nickname *pilot field* is computational, not quantum. This repository makes no de Broglie-Bohm claim about the brain.

## Why this is worth testing

Human theta/alpha oscillations can form cortical traveling waves with measurable spatial phase gradients, and recent work reports behavior-dependent changes in wave direction. That makes wave geometry a real signal-processing object. It does **not** imply that alpha is a master controller.

The new code therefore tests prediction, not metaphysics.

## P0 — instrument sanity

`tests/test_pilot_field_metrics.py` constructs a known planar traveling phase field and checks that the recovered propagation proxy points in the correct direction. A second synthetic packet test checks that the future-motion alignment statistic beats circular time-shift surrogates.

Passing P0 means only that the meter can recover the synthetic quantity it was designed to measure.

## P1 — slow field -> future fast trajectory

`pilot_field_explorer.py`:

1. loads ordinary EEG with MNE;
2. filters a prespecified slow phase band (default alpha 8-12 Hz);
3. filters a prespecified faster band (default 30-45 Hz);
4. estimates a local 2-D phase gradient from neighboring positioned EEG sensors;
5. converts `-grad(phi)` into a propagation-direction proxy;
6. computes the spatial centroid of relative fast-band analytic amplitude;
7. asks whether current slow-wave direction aligns with the centroid's future displacement;
8. compares the observed alignment with large circular time shifts;
9. runs a blocked-CV prediction test: current fast dynamics alone versus current fast dynamics + slow-wave direction.

A positive result needs **both** a surrogate advantage and a predictive advantage. Even then, sensor-space volume conduction and source mixing remain alternative explanations.

## P2 — fast pattern -> future slow-field change

The same run also reports a blocked-CV `writeback_predictive_gain`:

- baseline: future phase-flow direction from current phase-flow direction;
- enhanced: add current fast centroid and global fast amplitude.

Positive gain means the fast pattern contains incremental predictive information. It does **not** prove that fast activity writes the slow field.

## Stop rules

Do not rescue a null result by scanning bands, regions, lags, references, smoothing constants, or source models until something turns positive. A serious EEG result should freeze those choices first and then replicate on held-out recordings or subjects.

Do not interpret a sensor-space phase gradient as cortical propagation without confronting reference effects, volume conduction, source mixing and spatial sampling.

Do not call PAC, alignment, or predictive gain a causal conductor.

## Relationship to the historical code

The old source-space `abs(exp(i*dphi))` metric is algebraically constant at 1. The old min-to-max orchestra filter could also put the excluded conductor band back into the orchestra. Those maps are retained as provenance, not evidence.

`conductor_metrics.py` and `mnebrain_conductor_pac.py` remain the repaired PAC association branch. `pilot_field_metrics.py` is a different test: **spatial propagation geometry -> future activity trajectory**.
