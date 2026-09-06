# Regional Attractor Explorer

## 2026 reset: keep the instrument, kill the story

This repository began as an EEG visualization experiment around regional alpha power, multi-band composites and 3-D state-space trajectories. An old AI-written branch inflated that into a **"Universal Brain Coordination Model"** with a conductor/orchestra story.

That strong story is not supported. One historical coordination metric was also algebraically broken.

The repo now treats its history as an audit trail and asks smaller questions that can fail.

**Live site:** https://anttiluode.github.io/RegionalAttractorExplorer/

## What died

The historical source-space code computed

```python
plv_instantaneous = np.abs(np.exp(1j * (phase_conductor - phase_moire)))
```

but `|exp(i theta)| = 1` for every phase angle. The old pointwise "PLV" therefore contained no phase-locking information, its phase-slip value collapsed to zero, and `Y*PLV` collapsed to `Y`.

A second bug built the "orchestra" by filtering continuously from the minimum selected frequency to the maximum selected frequency. With alpha excluded as the candidate conductor, that broad filter could simply include alpha again.

**Old coordination screenshots are provenance, not evidence.** See [`docs/CONDUCTOR_AUDIT_2026.md`](docs/CONDUCTOR_AUDIT_2026.md).

## What survived

- `regional_attractor_explorer.py` remains a feature-space visualization tool. A trajectory can be useful without being a literal neural attractor.
- `conductor_metrics.py` / `mnebrain_conductor_pac.py` repair the old algebra with conventional windowed PAC. PAC is association, not causal control.
- Spatial traveling-wave geometry is a legitimate measurement target.
- The useful habit that survived everything else is **gates, receipts, nulls and stop rules**.

## Pilot-field audit

"Pilot field" is a computational nickname, **not a de Broglie-Bohm or quantum-brain claim**.

The first-pass question was deliberately simple:

> Does the phase geometry of a slow oscillation contain information about where faster activity moves next?

The frozen P1 screen was:

```text
8-12 Hz analytic phase
        ↓
neighbor-based spatial phase gradient
        ↓
phase-flow direction proxy
        ↓
future displacement of 30-45 Hz amplitude centroid, 80 ms later
        ↓
circular-shift null + blocked-CV predictive gain
```

Run one file:

```bash
python pilot_field_explorer.py recording.edf --start 0 --duration 60
```

Run a cohort:

```bash
python pilot_field_batch.py "*.edf" --start 0 --duration 60 --out results/pilot_field_batch.json
```

A useful P1 result was required to show both a shift-null alignment advantage **and** positive held-out predictive gain.

## Negative-result wall

| Gate / dataset | Result |
|---|---|
| Historical pointwise conductor PLV | **FAIL** — algebraically constant |
| Corrected PAC instrument | synthetic known-answer **PASS** |
| External phase-timed iEEG Gate Q | **FAIL — `NO_QUERY_WINDOW_ADVANTAGE`** |
| Pilot P0 synthetic traveling-wave meter | **PASS** |
| First two 64-channel EEG recordings | **`P1_FIRST_TWO_RECORDINGS_NULL`** |
| PhysioNet S016, 14 repeated runs from one person | **`P1_S016_WHOLE_RUN_NULL`** |
| Healthy cohort, 14 separate subjects | **`P1_HEALTHY_14_SUBJECT_REPLICATION_NULL`** |
| Schizophrenia cohort, 14 separate subjects | **`P1_SCHIZOPHRENIA_14_SUBJECT_REPLICATION_NULL`** |
| Healthy vs schizophrenia frozen group gate | **`NO_GROUP_DIFFERENCE_IN_P1_PREDICTIVE_GAIN`** |
| P2 writeback score | **`P2_WRITEBACK_ESTIMATOR_UNSTABLE`** |
| S016 event-conditioned task-vs-rest gate | **OPEN — separately frozen** |

The simple whole-run passive P1 mapping has therefore been repeatedly rejected. That does **not** mean traveling waves are absent. It means this specific sensor-space mapping has not earned support:

> **8–12 Hz phase-flow → future 30–45 Hz amplitude-centroid motion at 80 ms.**

## 28-subject frozen patient-control gate

The patient-control statistic was frozen before the schizophrenia cohort was inspected.

Primary endpoint: subject-level `guidance_predictive_gain`.

Primary test: two-sided subject-label permutation test, 100,000 permutations.

| endpoint | healthy (n=14) | schizophrenia (n=14) | SZ − H | p |
|---|---:|---:|---:|---:|
| guidance predictive gain | -0.002371 | -0.002856 | -0.000484 | **0.6884** |
| guidance alignment | +0.001872 | -0.000249 | -0.002121 | **0.6751** |

Healthy predictive gain was negative in 11/14 subjects. Schizophrenia predictive gain was negative in 14/14 subjects. A small negative cross-validated gain is compatible with adding an irrelevant predictor: it can slightly worsen held-out prediction without implying an "anti-guidance" mechanism.

**Verdict: `NO_GROUP_DIFFERENCE_IN_P1_PREDICTIVE_GAIN`.**

See [`docs/PILOT_FIELD_SZ_GROUP_GATE.md`](docs/PILOT_FIELD_SZ_GROUP_GATE.md) and [`results/pilot_field_SZ_group_gate.json`](results/pilot_field_SZ_group_gate.json).

## P2/writeback is not ready

The current reverse-direction score asks whether fast spatial state improves prediction of future slow-flow state. Its present relative-MSE formulation is numerically fragile.

- healthy `h14`: `-0.45241`
- schizophrenia `s07`: `-5.07896`
- cohort medians remain near `-0.003`

That pattern is a meter warning, not a biological result. The current classification is **`P2_WRITEBACK_ESTIMATOR_UNSTABLE`**. Do not use it for group claims until a replacement score is specified and validated before new outcomes are inspected.

## Event-conditioned gate remains open

Some EDF+ datasets contain predefined task annotations. `pilot_field_batch.py` records those annotations but intentionally ignores them for the frozen whole-run P1 screen.

`pilot_field_event_gate.py` asks a different, separately frozen question: does the **same fixed measurement** differ between predefined task and rest epochs? It does not reopen bands, lag, reference or region after seeing the nulls.

That is the next legitimate biological gate. Searching frequency/lag/region until something becomes significant is not.

## The hard EEG boundary

Sensor-space traveling-wave structure can be distorted by reference choice, volume conduction, source mixing, spatial sampling, filters and waveform shape. Scalp beta/gamma is also vulnerable to EMG.

The stop rule is explicit:

> **Do not rescue a null by searching bands, regions, lags, references, source models or smoothing choices until something becomes significant. Freeze the measurement first. Replicate second. Interpret last.**

## Recent empirical anchors

- Mohan et al., *Nature Human Behaviour* (2024), **The direction of theta and alpha travelling waves modulates human memory processing** — https://doi.org/10.1038/s41562-024-01838-3
- Koller et al., *Nature Communications* (2024), **Human connectome topology directs cortical traveling waves and shapes frequency gradients** — https://doi.org/10.1038/s41467-024-47860-x
- Model-based MEG/EEG traveling-wave recovery in human visual cortex (2025) — https://pubmed.ncbi.nlm.nih.gov/40245091/
- Traveling waves linking visual and frontal cortex during memory-guided behavior (2025) — https://pubmed.ncbi.nlm.nih.gov/40699921/

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

Synthetic meter test:

```bash
python -m pytest -q tests/test_pilot_field_metrics.py
```

## License

MIT.
