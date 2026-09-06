# Pilot-field first real-data receipt

Two 60-second, 64-channel EEG recordings were run through the frozen first-pass screen:

```text
phase band: 8-12 Hz
fast band: 30-45 Hz
future lag: 80 ms
average reference
sensor-space phase-gradient fit
200 circular-shift surrogates
blocked held-out prediction
```

The files were not used to tune the bands, lag or metric.

## Results

| file | phase-fit coherence | guidance alignment | shift-null p | null z | guidance predictive gain | writeback predictive gain |
|---|---:|---:|---:|---:|---:|---:|
| `2.edf` | 0.4846 | +0.00583 | 0.3781 | +0.271 | -0.00494 | +0.00094 |
| `3.edf` | 0.4980 | -0.02067 | 0.9055 | -1.373 | -0.00136 | -0.00249 |

The phase-fit coherence values say that the neighboring-sensor phase differences are not completely structureless under the local planar fit. They do **not** by themselves establish a cortical traveling wave.

The proposed guidance effect is absent in both recordings. `2.edf` has essentially zero alignment and does not beat time-shifted pairings. `3.edf` is slightly anti-aligned and is worse than most of its positive-direction shift null.

More importantly, adding the current slow-wave flow direction to a held-out baseline containing current fast-centroid position and recent displacement does not improve prediction in either recording. The gain is negative in both files.

The reverse `fast pattern -> future slow flow` screen also has no convincing signal. `2.edf` is positive by only 0.00094 (~0.09% relative MSE improvement) and there is no surrogate/significance layer for this metric yet; `3.edf` is negative.

## Classification

**`P1_FIRST_TWO_RECORDINGS_NULL`**

and, provisionally,

**`P2_NO_EVIDENCE_IN_FIRST_TWO_RECORDINGS`**

This does not prove that slow traveling waves never guide faster activity. It kills the narrower first-pass claim that the frozen 8-12 Hz sensor-space phase-flow estimate at an 80 ms lag gives a detectable, general guidance signal in these two recordings.

## Stop rule

Do not search bands, lags, regions, references, source models, centroid sharpenings or smoothing settings on these same two files and then relabel a surviving combination as confirmation.

A legitimate next step must be one of:

1. apply the exact frozen screen to more independent recordings;
2. predeclare a different mechanistic test before inspecting its outcomes; or
3. improve the measurement for a known methodological reason and rerun on held-out data.

The attractive story remains downstream of the measurement.
