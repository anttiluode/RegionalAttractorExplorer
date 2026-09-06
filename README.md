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

- `regional_attractor_explorer.py` remains a useful feature-space visualization tool. A trajectory can be useful without being a literal neural attractor.
- `conductor_metrics.py` / `mnebrain_conductor_pac.py` repair the old algebra with conventional windowed PAC. PAC is association, not causal control.
- The external phase-timed iEEG Gate Q returned **`NO_QUERY_WINDOW_ADVANTAGE`** for the stronger claim that theta-synchronized stimulation expands future response dimensionality. That null stays.
- Spatial traveling-wave geometry is a legitimate measurement target, so the repo now tests it directly and predictively.

## Pilot-field audit

"Pilot field" is a computational nickname, **not a de Broglie-Bohm or quantum-brain claim**.

The questions are:

> **P1:** Does the phase geometry of a slow oscillation contain information about where faster activity moves next?

> **P2:** Does the current fast-activity pattern contain information about how the slow phase field changes next?

The frozen first-pass P1 screen is:

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

The primary logic was fixed before real-data replication: a useful P1 result should show both a shift-null alignment advantage **and** positive held-out predictive gain.

## Current receipts

| Gate / dataset | Result |
|---|---|
| Historical pointwise conductor PLV | **FAIL** — algebraically constant |
| Corrected PAC instrument | synthetic known-answer **PASS** |
| External phase-timed iEEG Gate Q | **FAIL — `NO_QUERY_WINDOW_ADVANTAGE`** |
| Pilot P0 synthetic traveling-wave meter | **PASS** |
| First two 64-channel EEG recordings | **`P1_FIRST_TWO_RECORDINGS_NULL`** |
| PhysioNet S016, 14 repeated runs from one person | **`P1_S016_WHOLE_RUN_NULL`** |
| Healthy cohort, 14 separate subjects | **`P1_HEALTHY_14_SUBJECT_REPLICATION_NULL`** |
| S016 event-conditioned task-vs-rest gate | open / separately frozen |
| Healthy-vs-schizophrenia subject-level group gate | open / frozen before patient results |

### Healthy 14-subject replication

Frozen settings were unchanged: first 60 s, 8–12 Hz phase field, 30–45 Hz fast field, 80 ms lag.

- mean guidance alignment: `+0.001872`
- median guidance alignment: `-0.000214`
- positive alignments: `7 / 14`
- nominal alignment `p < .05`: `1 / 14`
- mean guidance predictive gain: `-0.002371`
- median guidance predictive gain: `-0.002135`
- positive predictive gain: `3 / 14`
- subjects satisfying both intended P1 requirements: **`0 / 14`**

The one nominal alignment hit (`h10`, `p=0.0348`) had **negative** held-out predictive gain, so it does not pass P1.

See [`docs/PILOT_FIELD_HEALTHY_COHORT.md`](docs/PILOT_FIELD_HEALTHY_COHORT.md) and [`results/pilot_field_SZ_healthy_summary.json`](results/pilot_field_SZ_healthy_summary.json).

## Frozen patient-control gate

The schizophrenia files have not yet been used to choose the group statistic. The group comparison is now frozen in [`pilot_field_group_gate.py`](pilot_field_group_gate.py):

- **primary:** subject-level `guidance_predictive_gain`
- **test:** two-sided subject-label permutation test
- **secondary:** `guidance_alignment`
- **writeback:** descriptive only in this gate

Writeback is deliberately not promoted to a primary group endpoint because healthy subject `h14` already produced an extreme negative value (`-0.45241`), revealing numerical fragility in that secondary score before patient results were inspected.

Once both cohort receipts exist:

```bash
python pilot_field_group_gate.py \
  results/pilot_field_SZ_healthy_batch.json \
  results/pilot_field_SZ_schizophrenia_batch.json \
  --out results/pilot_field_SZ_group_gate.json
```

A significant result would mean only that this **sensor-space screening metric differs between groups**. It would not establish causal wave guidance, a schizophrenia mechanism, consciousness, or quantum physics.

## Event-conditioned gate

Some EDF+ datasets contain task annotations. `pilot_field_batch.py` records those annotations but intentionally ignores them for the frozen whole-run P1 screen. `pilot_field_event_gate.py` is a separate gate for predefined task-vs-rest epochs; it does not reopen bands or lag after inspecting whole-run results.

## The hard EEG boundary

Sensor-space traveling-wave structure can be distorted by reference choice, volume conduction, source mixing, spatial sampling, filters and waveform shape. Scalp beta/gamma is also vulnerable to EMG.

So the stop rule is explicit:

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
