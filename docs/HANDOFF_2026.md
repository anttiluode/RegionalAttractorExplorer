# HANDOFF 2026 — RegionalAttractorExplorer

## One-sentence state

The historical source-space "conductor" metric failed audit; the surviving branch is a corrected PAC instrument plus a preregistered external **query-window** test on phase-locked human iEEG stimulation.

## Read in this order

1. `CONDUCTOR_AUDIT_2026.md` — what died and what survived.
2. `GATE_Q_THETA_QUERY_WINDOW.md` — frozen external question and kill criterion.
3. `GATE_Q_DATA_RECEIPT.md` — exact public dataset/within-subject pilot receipt.
4. `conductor_metrics.py` — audited signal metrics.
5. `mnebrain_conductor_pac.py` — corrected GUI branch; inherits Standard Analysis from the old app.

## What not to claim

- A PAC hotspot is not a conductor.
- A phase effect is not consciousness.
- The corrected screenshot is not a result until it beats nulls.
- The Kragel paper's published connectivity result does not automatically validate Gate Q.
- Geometry/effective-rank language must beat amplitude/artifact/channel-count controls.

## Current hard receipts

- Historical pointwise `abs(exp(i*dphi))` is algebraically 1 and therefore unusable as PLV.
- The historical orchestra min-to-max filter can reintroduce the excluded conductor band.
- Corrected PAC separates a known synthetic modulated signal from a matched control.
- OpenNeuro `ds006065` contains a strong within-subject pilot: p17 and p19 each have both theta-synchronized and phase-blind closed-loop recordings.
- The four raw `.eeg` payloads for that pilot total ~3.38 GB, so full-dataset download is unnecessary initially.

## Immediate next action

Fetch only p17/p19 `cl` and `clcontrol` BrainVision files from OpenNeuro/DataLad, extract stimulation events, and implement Gate Q exactly as frozen.

Primary response window: 50-250 ms. Secondary: 15-50 ms. Baseline: -50 to -10 ms. Primary comparison is within subject TS vs PB with effective-rank/SV-entropy, amplitude-normalized controls, trial permutations, temporal shifts, pre-stim pseudo-windows, and artifact diagnostics.

If it fails, record `NO_QUERY_WINDOW_ADVANTAGE` and do not rescue with new bands/regions/embeddings.
