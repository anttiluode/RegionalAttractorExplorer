# Conductor audit 2026

The old "conductor frequency" branch is reopened only as a falsifiable slow/fast question.

> **Can a slow state define a temporary control surface that changes which fast trajectories are reachable?**

The name *conductor* is retained as a nickname. Coupling is not causal control.

## Historical failure

The source-space code in `mnebrain_signalvs_composite3.py` used:

```python
plv_instantaneous = np.abs(np.exp(1j * (phase_conductor - phase_moire)))
```

But `|exp(i theta)| = 1` for every phase angle. Therefore the historical `Phase-Slip Rate` was zero and `Coordinated Power (Y*PLV)` collapsed to orchestra power. The old coordination map is not evidence of coordination.

A second bug selected non-conductor bands and then applied one continuous min-to-max bandpass. With alpha as conductor and delta/theta/beta/gamma selected as orchestra, this spans roughly 0.5-50 Hz and re-introduces alpha.

Keep both failures visible.

## Repaired screen

`conductor_metrics.py` implements windowed phase-amplitude coupling (PAC): slower candidate phase versus faster-band analytic amplitude. Orchestra bands are filtered independently, only faster selected bands are used, and no post-hoc band weights are tuned in the audit branch.

PAC is established signal-processing machinery and is treated here only as association. A real EEG result requires prespecified temporal/spectral nulls.

## Known-answer sanity test

The synthetic unit test uses a 10 Hz phase signal and matched noisy 40 Hz carriers, one explicitly amplitude-modulated and one unmodulated.

Current receipt:

```text
new PAC, phase-modulated signal : 0.3694
new PAC, matched control        : 0.0120
historical |exp(i*dphi)| max |x-1|: 2.220e-16
PASS
```

This validates the instrument only. It is not a biological result.

## What the new screenshot means

The corrected `Y*PAC` source visualization is now algebraically capable of varying in space/time, unlike the historical constant-PLV branch. Hotspots still earn nothing without nulls because source leakage, waveform shape, transients, common input, and filtering can create convincing PAC structure.

## Better external test

Rather than interpret passive EEG hotspots first, use the intervention dataset preregistered in `GATE_Q_THETA_QUERY_WINDOW.md`: Kragel et al. 2025 / OpenNeuro `ds006065`.

The strongest first pilot is within-subject TS vs phase-blind stimulation in p17 and p19. The question is whether phase-timed intervention changes future-response geometry, not whether PAC is nonzero.

## Biological motivation, not evidence

Leterrier (2018) summarizes a genuine slow/fast boundary at the axon initial segment: fast channel modulation can occur over seconds-minutes, slower morphological plasticity over hours-days, and diminished activity can lengthen the AIS while elevated activity can shift it distally. This motivates the abstraction of slow variables editing a fast control surface; it does not establish the AI architecture.

## Stop rule

If the result needs hand-picked frequencies, regions, delays, nonlinear embeddings, or post-hoc windows to survive, stop. Preserve `NO_QUERY_WINDOW_ADVANTAGE` as a legitimate outcome.
