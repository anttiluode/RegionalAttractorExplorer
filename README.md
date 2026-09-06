# Regional Attractor Explorer

## 2026 reset: keep the instrument, kill the story

This repository started as an EEG visualization experiment around regional alpha power, multi-band composites and 3-D state-space trajectories. An old AI-written branch then inflated that into a **"Universal Brain Coordination Model"** with a conductor/orchestra story.

That strong story is not supported. Worse: one historical coordination metric was algebraically broken.

The repo now treats its history as an audit trail and asks smaller questions that can fail.

**Live site:** https://anttiluode.github.io/RegionalAttractorExplorer/

## What died

The historical source-space code computed

```python
plv_instantaneous = np.abs(np.exp(1j * (phase_conductor - phase_moire)))
```

but

```text
|exp(i theta)| = 1
```

for every phase angle. Therefore the old pointwise "PLV" contained no phase-locking information, its phase-slip value collapsed to zero, and `Y*PLV` collapsed to `Y`.

A second bug built the "orchestra" by filtering continuously from the minimum selected frequency to the maximum selected frequency. With alpha excluded as the candidate conductor, that broad filter could simply include alpha again.

**Old coordination screenshots are provenance, not evidence.** See [`docs/CONDUCTOR_AUDIT_2026.md`](docs/CONDUCTOR_AUDIT_2026.md).

## What survived

Three things remain useful:

1. **RegionalAttractorExplorer itself** is still a visualization tool for ordinary EEG-derived features. A trajectory in a chosen feature space can be useful without being a literal neural attractor.
2. [`conductor_metrics.py`](conductor_metrics.py) and [`mnebrain_conductor_pac.py`](mnebrain_conductor_pac.py) repair the old algebra with conventional windowed phase-amplitude coupling (PAC). PAC is association, not causal control.
3. The repository already made external contact with a stronger hypothesis. The preregistered phase-timed iEEG **Gate Q** returned **`NO_QUERY_WINDOW_ADVANTAGE`** for the claim that theta-synchronized stimulation expands future response dimensionality. That null stays. See [`docs/HANDOFF_2026.md`](docs/HANDOFF_2026.md).

## New branch: the pilot-field audit

"Pilot field" is a computational nickname, **not a de Broglie-Bohm or quantum-brain claim**.

The question is now spatial and predictive:

> **Does the phase geometry of a slow oscillation contain information about where faster activity moves next?**

and separately:

> **Does the current fast-activity pattern contain information about how the slow phase field changes next?**

Human theta/alpha traveling waves are a real empirical phenomenon, and recent work links wave direction to behavior. That makes the geometry worth measuring. It does not make alpha a master conductor.

### P0 — known-answer test

[`pilot_field_metrics.py`](pilot_field_metrics.py) estimates a slow-wave spatial phase gradient from neighboring positioned sensors and turns `-grad(phi)` into a propagation-direction proxy. [`tests/test_pilot_field_metrics.py`](tests/test_pilot_field_metrics.py) checks that it recovers a known synthetic traveling wave and that a synthetically guided activity packet beats circular-shift pairings.

```bash
python -m pytest -q tests/test_pilot_field_metrics.py
```

P0 validates the meter only.

### P1 — slow field -> future fast trajectory

[`pilot_field_explorer.py`](pilot_field_explorer.py) loads an EEG file and performs a conservative sensor-space screen:

```text
slow band (default 8-12 Hz)
        ↓ analytic phase
spatial phase gradient across neighboring sensors
        ↓
phase-flow direction proxy
        ↓
compare with future displacement of fast-band amplitude centroid
        ↓
circular-shift null + blocked-CV predictive gain
```

Run:

```bash
python pilot_field_explorer.py recording.edf --start 0 --duration 60
```

The default fast band is 30-45 Hz and the default future lag is 80 ms. Those defaults are hypotheses, not optimized findings.

A useful P1 result should show both:

- alignment stronger than prespecified large circular time shifts; and
- held-out prediction improvement when slow-wave direction is added to a baseline containing current fast trajectory dynamics.

### P2 — fast pattern -> future slow-field change

The same run reports `writeback_predictive_gain`:

```text
baseline: current slow flow -> future slow flow
enhanced: baseline + current fast centroid + fast global amplitude
```

Positive gain means incremental predictive information. It **does not** establish that the fast activity physically writes the slow field.

See [`docs/PILOT_FIELD_2026.md`](docs/PILOT_FIELD_2026.md).

## The hard EEG boundary

A sensor-space traveling-wave result can be convincing and still be an artifact of reference choice, volume conduction, source mixing, spatial sampling, filtering or waveform shape. Scalp beta/gamma is also vulnerable to EMG.

So the stop rule is explicit:

> Do not rescue a null by searching bands, regions, lags, references, source models or smoothing choices until something becomes significant.

Freeze the measurement first. Replicate second. Interpret last.

## Recent empirical anchors

- Mohan et al., *Nature Human Behaviour* (2024), **The direction of theta and alpha travelling waves modulates human memory processing**: https://doi.org/10.1038/s41562-024-01838-3
- Koller et al., *Nature Communications* (2024), **Human connectome topology directs cortical traveling waves and shapes frequency gradients**: https://doi.org/10.1038/s41467-024-47860-x
- Model-based MEG/EEG traveling-wave recovery in human visual cortex (2025): https://pubmed.ncbi.nlm.nih.gov/40245091/
- Traveling waves linking visual and frontal cortex during memory-guided behavior (2025): https://pubmed.ncbi.nlm.nih.gov/40699921/
- Kragel et al., *Nature Communications* (2025), closed-loop theta stimulation used by this repo's Gate Q: https://doi.org/10.1038/s41467-025-59417-7

## Historical tools kept on purpose

- `regional_attractor_explorer.py` — original regional 3-D feature explorer.
- `mnebrain_signalvs_composite3.py` — historical source-space branch; **do not use its old pointwise PLV as evidence**.
- `AI_brainstate_analyzer.py`, `ai_autoencoder_signal_analysis.py`, `gamma_gating_explorer_for_temporal_lobes.py` — exploratory historical branches. Their prose conclusions are hypotheses/provenance unless separately audited.
- `mnebrain_conductor_pac.py` — repaired PAC association screen.
- `gate_q_external.py`, `gate_q_external_grid.py` — frozen external Gate Q analysis and event-coordinate correction.

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

For only the synthetic pilot-field tests:

```bash
pip install numpy scipy pytest
python -m pytest -q tests/test_pilot_field_metrics.py
```

## License

MIT.
