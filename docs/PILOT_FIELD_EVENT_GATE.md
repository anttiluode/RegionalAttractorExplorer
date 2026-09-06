# P1E — event-conditioned pilot-field gate

## Status

**Frozen before event-conditioned outcomes are inspected.**

This is a new hypothesis motivated by the experimental design of the PhysioNet EEG Motor Movement/Imagery dataset. It is not a rescue of the whole-run P1 result.

## Question

> Does the predefined task state (`T1` or `T2`) strengthen the relationship between the 8-12 Hz spatial phase-flow proxy and future 30-45 Hz activity motion compared with predefined rest (`T0`)?

## Dataset scope

Primary pilot: subject `S016`, runs `R03` through `R14`.

Runs `R01` and `R02` are baseline eyes-open / eyes-closed recordings and are excluded from the primary event contrast because they do not alternate task and rest annotations.

The task runs contain repeated `T0`, `T1`, and `T2` annotations. For the primary gate:

- `REST` = `T0`
- `TASK` = `T1` or `T2`

The primary gate does **not** distinguish left/right, fists/feet, executed/imagined, or individual run families. Those are secondary future questions and must not be searched to rescue a null primary result.

## Frozen signal settings

Keep the P1 measurement unchanged:

- phase band: **8-12 Hz**
- fast band: **30-45 Hz**
- lag: **80 ms**
- average reference
- same 64-channel sensor geometry
- same neighbour phase-gradient estimator
- same fast-amplitude centroid definition

Use the full task recordings so every annotated block is available. Filtering is performed on the continuous recording before event selection.

## Event window rule

For each annotation block, discard **0.5 s** from the beginning and **0.5 s** from the end. Also require both the previous-lag and future-lag samples used by the predictive model to remain inside the same annotation block.

This guard is fixed before outcomes to reduce transition/filter-boundary contamination without tuning the window after seeing results.

## Per-run measurements

For every run `R03-R14`, compute separately for `TASK` and `REST`:

1. guidance alignment;
2. held-out guidance predictive gain.

Cross-validation must leave whole annotation blocks out rather than randomly mixing adjacent samples between train and test.

Define per-run contrasts:

```text
Delta_alignment = alignment_TASK - alignment_REST
Delta_predictive = predictive_gain_TASK - predictive_gain_REST
```

## Primary inference

There are 12 task runs from the same participant. Treat them as repeated-run evidence, **not 12 independent people**.

Use an exact one-sided sign-flip test across the 12 run-level deltas for each endpoint. With 12 runs there are only `2^12 = 4096` sign configurations, so no Monte Carlo approximation is needed.

## Pass rule

P1E passes only if **both** are true:

1. mean `Delta_alignment > 0` with exact sign-flip `p < 0.05`; and
2. mean `Delta_predictive > 0` with exact sign-flip `p < 0.05`.

If only one endpoint passes, classify the gate as **inconclusive / partial**, not as evidence for a pilot field.

If neither passes, classify:

**`NO_EVENT_CONDITIONED_GUIDANCE_ADVANTAGE`**

## Stop rule

After the outcome is known, do not change bands, lag, reference, scalp region, centroid sharpening, guard duration, or task grouping to obtain significance.

Executed-vs-imagined, left-vs-right and fists-vs-feet comparisons may be interesting, but they are explicitly secondary and cannot retroactively rescue the primary TASK-vs-REST gate.
