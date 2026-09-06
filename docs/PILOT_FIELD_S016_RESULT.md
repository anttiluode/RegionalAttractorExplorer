# Pilot-field S016 batch receipt

## Frozen screen

This receipt records the first full 14-run batch on PhysioNet EEG Motor Movement/Imagery subject `S016` using the already-frozen P1 settings:

- slow phase band: **8-12 Hz**
- fast amplitude band: **30-45 Hz**
- future lag: **80 ms**
- analysis window: first **60 s** of each run
- average reference, full positioned 64-channel sensor array
- 200 circular-shift surrogates
- event labels recorded but **not used**

All 14 files completed successfully.

## Batch summary

| quantity | result |
|---|---:|
| mean guidance alignment | +0.001357 |
| mean guidance predictive gain | -0.000738 |
| mean writeback predictive gain | -0.005451 |
| nominal shift-null p < .05 | 1 / 14 |
| runs passing the original P1 conjunction | **0 / 14** |

The original P1 screen required both stronger-than-shifted alignment and positive held-out predictive gain. No S016 run satisfied both.

`S016R07.edf` produced the only nominal alignment result below .05 (`alignment=+0.03000`, `p=0.04478`, `z=2.0347`), but its held-out guidance predictive gain was **negative** (`-0.00416`). It therefore does not pass P1. `S016R14.edf` was near the nominal alignment threshold (`p=0.05473`) but also had negative predictive gain.

## Important interpretation boundary

These are **14 runs from one participant**, not 14 independent participants. The repetition is useful for testing whether the effect is stable within one person across sessions/tasks, but it is not a population-level sample size of 14.

The mean phase-gradient fit coherence was about **0.638**, so the result is not simply "there was no measurable alpha spatial structure." The narrower result is:

> Under this sensor-space estimator, measurable 8-12 Hz spatial phase geometry did not provide stable predictive guidance for 30-45 Hz activity-centroid motion 80 ms later.

Writeback was even less encouraging: only 2/14 runs had positive writeback predictive gain and the batch mean was negative.

## Why one p < .05 does not rescue the hypothesis

With 14 nominal tests, seeing one p-value below .05 is not surprising by itself. More importantly, the repository's gate was deliberately conjunctive: a visually/numerically positive alignment is not enough if adding the slow-field vector makes held-out prediction worse.

Therefore the classification is:

**`P1_S016_WHOLE_RUN_NULL`**

and provisionally:

**`P2_S016_NO_EVIDENCE`**

## Event information is a new gate, not a rescue

Runs 3-14 contain alternating `T0` rest and `T1/T2` task annotations. The current batch deliberately ignored them. That matters because a whole-run average mixes task and rest states.

A separately frozen event-conditioned gate is specified in [`PILOT_FIELD_EVENT_GATE.md`](PILOT_FIELD_EVENT_GATE.md). It keeps the same bands, lag, sensor geometry and reference, and asks whether the already-defined task state changes the guidance relationship. It must be treated as a new hypothesis, not as post-hoc repair of P1.
