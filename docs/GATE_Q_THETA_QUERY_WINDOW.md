# Gate Q — theta query-window external test

## Status

**Preregistered before looking at raw trial-level outcomes.**

This branch does not claim that theta is a conscious command signal, that one rhythm is a universal conductor, or that phase coupling proves control. The narrow question is whether a low-dimensional timing state changes the set of future network responses produced by a matched intervention.

## External contact

Dataset: OpenNeuro `ds006065` v1.0.0, Kragel et al. (2025), *Closed-loop control of theta oscillations enhances human hippocampal network connectivity*.

The experiment is unusually useful for this repository because lateral temporal cortex stimulation was delivered either:

- **TS** — synchronized to the ongoing hippocampal theta trough; or
- **PB** — phase blind, at approximately the same stimulation rate.

This is much stronger contact than passive PAC: the experiment perturbs the system and supplies a phase-blind control.

## Public dataset audit

The BIDS mirror contains seven participants:

`p16, p17, p18, p19, p20, UC004, UC005`.

Condition availability from the public file tree:

| subject | TS `cl` | PB `clcontrol` | strongest use here |
|---|---:|---:|---|
| p16 | yes | no | TS-only expansion |
| p17 | yes | yes | **primary within-subject pilot** |
| p18 | yes | no | TS-only expansion |
| p19 | yes | yes | **primary within-subject pilot** |
| p20 | no | yes | PB-only expansion |
| UC004 | no | yes | PB-only expansion |
| UC005 | no | yes | PB-only expansion |

Therefore the first test is deliberately **p17 + p19**, because both conditions exist within the same brains/electrode geometries. Do not pool all seven first.

Metadata check:

- p17: 175 SEEG channels; nominal 500 Hz; TS duration ~3023 s; PB duration ~3428 s.
- p19: 168 SEEG channels; nominal 500 Hz; TS duration ~1570 s; PB duration ~1768 s.
- raw files are BrainVision and are git-annex-backed in the GitHub mirror; obtain content from OpenNeuro/DataLad rather than treating the symlinks as data.

## What the published result already says

The paper reports that TS, relative to PB:

- was genuinely phase locked to hippocampal theta;
- increased hippocampal theta during repetitive stimulation;
- increased late (50–250 ms) hippocampal stimulation-evoked response amplitude relative to PB;
- produced persistent post-stimulation increases in SEP and phase-lag-index connectivity.

Important resistance: the classic magnitude-squared coherence reanalysis did **not** show a significant condition or time-by-condition effect in theta. Imaginary coherence did reproduce the PLI direction. The connectivity story is therefore not metric-free.

Those are the authors' outcomes. Gate Q asks a different question and must not retro-fit its metric to reproduce them.

## Frozen question

> **At fixed intervention type and approximately fixed stimulation rate, does synchronizing the intervention to an endogenous low-dimensional theta phase produce a measurably different/richer geometry of future network responses than phase-blind intervention?**

This is the operational version of the `query-window` idea:

```text
small timing state
      +
matched intervention
      ↓
large distributed future response
```

The timing state is not assumed to contain the response.

## Primary response object

For stimulation event `i`, build a post-stimulus response vector from non-stimulation contacts:

`R_i = vec(response[channel, time])`.

Primary window: **50–250 ms** after stimulation.

Reason: this matches the paper's late SEP window, where indirect/network effects were reported, while avoiding the immediate stimulation boundary as much as possible.

Secondary preregistered window: **15–50 ms**. Treat this as the early/direct comparison, not a rescue window.

Baseline correction: **-50 to -10 ms**, matching the paper where possible.

Do not use samples at/around the stimulation artifact merely because they increase dimensionality.

## Primary geometry metrics

Compute all metrics on identical channel sets within a subject and with identical preprocessing in TS and PB:

1. **effective rank** of the trial × response-feature matrix;
2. **singular-value entropy**;
3. **cross-validated phase/condition separability** of response trajectories using a simple linear classifier;
4. optional descriptive PCA trajectory plots, never as evidence by themselves.

Report raw-amplitude and per-feature-normalized versions. This separates 'everything got bigger' from 'the response space changed shape'.

## Primary contrast

Within each of p17 and p19:

`Delta_Q = geometry(TS) - geometry(PB)`.

Primary evidence requires the same direction in **both** participants for the prespecified effective-rank/SV-entropy family, plus superiority to the matched null distribution below.

With only two within-subject participants this is a pilot/gate, not a population neuroscience claim.

## Mandatory nulls / boring controls

1. **trial-label permutation** within subject;
2. **circular temporal shift** of response epochs relative to stimulation events;
3. **pre-stimulus pseudo-response** with matched window length;
4. **channel-count match** and identical usable contacts between TS/PB;
5. **amplitude normalization** control;
6. **early-window artifact diagnostic**;
7. **negative-lag diagnostic** — any equally strong 'future' effect before stimulation is a warning;
8. **PB is the primary experimental control**. Surrogates do not replace it.

If stimulation artifact removal or event extraction differs by condition, stop and fix that before computing Gate Q.

## Kill criterion

Call:

`NO_QUERY_WINDOW_ADVANTAGE`

if the TS-vs-PB response-geometry difference in p17/p19 does not exceed the prespecified matched nulls, or if it disappears under amplitude normalization / artifact-safe windows.

Do not rescue by adding frequency bands, regions, nonlinear embeddings, or hand-picked response windows after seeing labels.

## Expansion only after the within-subject pilot

If p17 and p19 survive:

1. freeze the pipeline;
2. add TS-only p16/p18 and PB-only p20/UC004/UC005 using a subject-level model;
3. retain anatomy/electrode-coverage covariates and equalized feature counts;
4. ask whether the direction generalizes.

## Relation to PivotPoint / GeometricNeuron

This is an architectural analogy, not biological evidence for either repo.

- PivotPoint asks how a small current action changes what becomes reachable/readable next.
- Gate Q asks whether a small timing state changes which future network trajectories a fixed intervention can reach.
- GeometricNeuron asks whether physical structure creates additional effective degrees of control.

Common measurable object: **the effective dimensionality of materially different reachable futures**.

## Stop rule

If Gate Q becomes a story that every outcome can support, stop. The phase-blind arm and the nulls are here specifically to let the idea die.
