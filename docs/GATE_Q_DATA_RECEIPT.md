# Gate Q dataset receipt — OpenNeuro ds006065

This note records what is visible from the public BIDS/DataLad mirror before downloading raw trial-level signals.

## Strongest pilot subjects

The two participants with both theta-synchronized (`cl`) and phase-blind (`clcontrol`) recordings are:

- `sub-p17`
- `sub-p19`

Their scan manifests contain both closed-loop conditions plus pre/post evoked-potential and rest recordings for each condition. This makes them the primary within-subject Gate Q pilot.

## Recording metadata

| subject | condition | channels | nominal fs | duration |
|---|---|---:|---:|---:|
| p17 | TS `cl` | 175 SEEG | 500 Hz | 3023.015 s |
| p17 | PB `clcontrol` | 175 SEEG | 500 Hz | 3428.286 s |
| p19 | TS `cl` | 168 SEEG | 500 Hz | 1570.047 s |
| p19 | PB `clcontrol` | 168 SEEG | 500 Hz | 1768.179 s |

The channel TSVs themselves report ~499.667 Hz for p17 contacts; analysis code must use the recording header rather than hard-code exact sample timing.

## Raw payload sizes from git-annex keys

The `.eeg` pointers encode these payload sizes:

- p17 TS: 1,057,350,000 bytes
- p17 PB: 1,199,100,000 bytes
- p19 TS: 527,184,000 bytes
- p19 PB: 593,712,000 bytes

Primary pilot total: **3,377,346,000 bytes (~3.38 GB decimal)** plus small BrainVision headers/markers/metadata.

So we do not need the full dataset first.

## Minimal DataLad fetch

Example:

```bash
datalad clone https://github.com/OpenNeuroDatasets/ds006065.git ds006065
cd ds006065

datalad get \
  sub-p17/ieeg/sub-p17_task-cl_ieeg.* \
  sub-p17/ieeg/sub-p17_task-clcontrol_ieeg.* \
  sub-p19/ieeg/sub-p19_task-cl_ieeg.* \
  sub-p19/ieeg/sub-p19_task-clcontrol_ieeg.*
```

Also keep these local metadata files:

```text
sub-p17/ieeg/sub-p17_electrodes.tsv
sub-p17/ieeg/sub-p17_task-cl_channels.tsv
sub-p17/ieeg/sub-p17_task-clcontrol_channels.tsv
sub-p19/ieeg/sub-p19_electrodes.tsv
sub-p19/ieeg/sub-p19_task-cl_channels.tsv
sub-p19/ieeg/sub-p19_task-clcontrol_channels.tsv
```

## Published-result check

Before Gate Q, the paper already reports a mixed but meaningful set of receipts:

- theta-synchronized stimulation was strongly phase locked relative to PB;
- hippocampal theta increased during TS relative to PB;
- the **late 50-250 ms** SEP component increased for TS relative to PB, while the early 15-50 ms component did not show the same group difference;
- persistent post-stimulation SEP and PLI connectivity increased following TS;
- classic magnitude-squared coherence did **not** show significant theta condition/time-interaction effects, while imaginary coherence did reproduce the PLI direction.

That last result matters: not every connectivity estimator says yes.

Gate Q must therefore remain distinct from reproducing the paper. Its frozen target is response-space geometry/effective dimensionality under TS vs PB.

## Current status

- Historical instantaneous-PLV conductor metric: dead.
- Corrected PAC known-answer instrument: passes synthetic unit test.
- Corrected passive PAC screenshot: visually non-degenerate, **not yet a result** because nulls are absent.
- External Gate Q: preregistered, raw p17/p19 trial analysis not yet run.
