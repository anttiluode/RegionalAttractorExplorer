# Pilot-field healthy-cohort receipt

## Frozen measurement

This receipt keeps the first-pass P1 settings unchanged:

- first 60 s of each EDF
- slow phase field: 8–12 Hz
- fast amplitude field: 30–45 Hz
- future lag: 80 ms
- 200 circular-shift surrogates
- average-reference sensor-space geometry

The cohort contains 14 separate healthy recordings (`h01.edf` … `h14.edf`). Unlike the earlier S016 batch, these files are treated as separate subjects rather than repeated runs from one person.

## Result

| quantity | healthy cohort result |
|---|---:|
| n subjects | 14 |
| mean guidance alignment | +0.001872 |
| median guidance alignment | -0.000214 |
| positive alignments | 7 / 14 |
| nominal alignment p < .05 | 1 / 14 |
| mean guidance predictive gain | -0.002371 |
| median guidance predictive gain | -0.002135 |
| positive predictive gains | 3 / 14 |
| mean writeback predictive gain | -0.035280 |
| median writeback predictive gain | -0.003201 |
| positive writeback gains | 0 / 14 |

No subject satisfied the intended two-part P1 logic of both a positive shift-null alignment result and positive held-out predictive gain.

`h10.edf` produced the only nominal alignment p-value below .05 (`p = 0.0348`, alignment `+0.02571`) but its held-out guidance predictive gain was negative (`-0.00090`). It therefore does not pass P1.

## Classification

**`P1_HEALTHY_14_SUBJECT_REPLICATION_NULL`**

The prespecified sensor-space 8–12 Hz phase-flow estimate does not show reproducible predictive guidance of 30–45 Hz spatial activity 80 ms later in this healthy cohort.

This is narrower than saying that alpha traveling waves are absent or functionally irrelevant. The phase-fit coherence remained measurable across subjects, but the fitted field direction did not consistently contribute the predicted future-motion information.

## Writeback warning discovered before patient-group analysis

`h14.edf` produced `writeback_predictive_gain = -0.45241`, far outside the other healthy values. The sign is against the writeback hypothesis, not in its favor, but the magnitude reveals that the current writeback score can become numerically fragile for a subject.

Therefore the healthy-vs-schizophrenia gate freezes the following rule **before the schizophrenia results are inspected**:

- primary group endpoint: `guidance_predictive_gain`
- two-sided subject-label permutation test
- `guidance_alignment`: secondary
- `writeback_predictive_gain`: descriptive only for this group gate

See `pilot_field_group_gate.py`.

## Interpretation boundary

A later patient/control difference, if one appears, is only a difference in this screening metric. It would not by itself establish wave guidance, a schizophrenia mechanism, a consciousness mechanism, or a quantum pilot wave.
