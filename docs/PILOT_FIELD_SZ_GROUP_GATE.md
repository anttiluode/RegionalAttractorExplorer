# Pilot-field patient-control gate — frozen null

## Status

**Verdict: `NO_GROUP_DIFFERENCE_IN_P1_PREDICTIVE_GAIN`.**

This comparison was frozen before the schizophrenia cohort was inspected. The primary endpoint was subject-level `guidance_predictive_gain`, tested with a two-sided subject-label permutation test over 100,000 permutations.

## Frozen measurement

- first 60 s of each recording
- slow phase band: 8–12 Hz
- fast amplitude band: 30–45 Hz
- future lag: 80 ms
- sensor-space phase-gradient / fast-centroid screen
- 14 healthy subjects vs 14 schizophrenia subjects

No frequency, lag, region, reference or centroid setting was reopened for this comparison.

## Result

| endpoint | healthy | schizophrenia | difference (SZ - H) | permutation p |
|---|---:|---:|---:|---:|
| guidance predictive gain | -0.002371 | -0.002856 | -0.000484 | 0.6884 |
| guidance alignment | +0.001872 | -0.000249 | -0.002121 | 0.6751 |

Healthy predictive gain was negative in 11/14 subjects. Schizophrenia predictive gain was negative in 14/14 subjects. The groups therefore do not show evidence for the preregistered-style P1 group difference.

A slightly negative cross-validated gain is compatible with an irrelevant extra predictor: adding slow-flow features can make held-out prediction a little worse without implying an active anti-guidance mechanism.

## What this kills

For this measurement, these data do **not** support either of the following claims:

1. a general passive scalp-EEG rule in which 8–12 Hz phase-flow predicts where 30–45 Hz activity moves 80 ms later;
2. a schizophrenia-control difference in that P1 predictive rule.

This does not invalidate traveling waves themselves, state-dependent wave phenomena, source-space analyses, or other independently specified phase/amplitude relationships.

## P2/writeback warning

The current `writeback_predictive_gain` score is not robust enough for group inference. Healthy `h14` produced `-0.45241`; schizophrenia subject `s07` produced `-5.07896`. The cohort medians are near `-0.003`, showing that the relative-gain score can become dominated by a nearly vanishing baseline MSE.

Classification: **`P2_WRITEBACK_ESTIMATOR_UNSTABLE`**. Keep it descriptive until a replacement score and its validation are specified before looking at new outcomes.

## What remains open

The S016 event-conditioned gate was frozen separately before its result was inspected. It asks a different question: whether the same fixed wave-guidance measurement changes between predefined task and rest epochs. That gate is not a rescue by frequency/lag search; it is an experimental-state test defined from external event markers.

## Interpretation boundary

A positive group result would only have meant a difference in this sensor-space screening metric. The observed null therefore says nothing about a quantum-brain theory, consciousness, or a causal mechanism of schizophrenia.

Raw frozen receipt: [`../results/pilot_field_SZ_group_gate.json`](../results/pilot_field_SZ_group_gate.json).
