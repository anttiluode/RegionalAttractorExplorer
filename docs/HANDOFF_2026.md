# HANDOFF 2026 — RegionalAttractorExplorer

## One-sentence state

The historical source-space "conductor" metric failed audit, the corrected PAC instrument survives as a visualization/association tool, and the preregistered external **Gate Q** on phase-locked human iEEG stimulation returned **`NO_QUERY_WINDOW_ADVANTAGE`** for the specific claim that theta-synchronized timing expands reachable response dimensionality.

## Read in this order

1. `CONDUCTOR_AUDIT_2026.md` — what died and what survived.
2. `GATE_Q_THETA_QUERY_WINDOW.md` — frozen external question and kill criterion.
3. `GATE_Q_DATA_RECEIPT.md` — exact public dataset/within-subject pilot receipt.
4. `GATE_Q_RESULT.md` — the frozen p17/p19 result.
5. `GATE_Q_AUTHOR_CORE.md` and `GATE_Q_AUTHOR_TIMESTAMPS.md` — authors' released preprocessing/event logic used to audit our event reconstruction.
6. `gate_q_external.py` — frozen analysis.
7. `gate_q_external_grid.py` — mechanical event-coordinate correction for the concatenated 3 s BIDS epochs; downstream gate is unchanged.
8. `conductor_metrics.py` / `mnebrain_conductor_pac.py` — corrected passive PAC instrument.

## What not to claim

- A PAC hotspot is not a conductor.
- A phase effect is not consciousness.
- The corrected screenshot is not a causal result.
- The Kragel paper's published connectivity/amplitude effects do not imply increased response dimensionality.
- A classifier separating TS from PB is not evidence that TS has more effective control degrees of freedom.
- Do not rescue Gate Q by adding hand-picked bands, regions, embeddings, windows, or geometry metrics after seeing this result.

## Historical conductor audit

- Pointwise `abs(exp(i*dphi))` is algebraically 1 and therefore unusable as PLV.
- The old orchestra min-to-max filter can reintroduce the excluded conductor band.
- Corrected windowed PAC separates a known synthetic modulated signal from a matched control.
- Passive PAC remains association only and requires nulls before interpretation.

## Gate Q external contact

Dataset: OpenNeuro `ds006065`, Kragel et al. 2025.

Primary within-subject pilot: p17 and p19, the two participants with both theta-synchronized (`cl`) and phase-blind (`clcontrol`) stimulation.

Frozen primary response: non-stimulation-contact time-domain response geometry from **50-250 ms** after the pulse, baseline **-50 to -10 ms**, with raw and pooled-per-feature normalized effective rank / singular-value-energy entropy. Negative-lag pseudo-response: **-250 to -50 ms**. Trial-label permutation nulls were fixed before outcomes.

## Event-recovery correction

The first raw run attempted to rediscover stimulation events from artifacts and failed the phase checksum for p19. That run is preserved in git history.

After the failure, inspection of the authors' source plus the public binary dimensions showed that the OpenNeuro BrainVision runs are already-epoched **3 s trials concatenated back-to-back**:

- authors: 1000 ms pre + 2000 ms post;
- every public pilot payload: exactly **1500 samples per expected trial**;
- therefore stimulation is deterministically sample **500** of each trial.

`gate_q_external_grid.py` changes only this event-coordinate plumbing. The response windows, metrics, normalization, nulls, seed and kill criterion were not changed.

The corrected event grid reproduces the intended phase manipulation strongly:

| subject | TS pre-stim theta phase consistency | PB |
|---|---:|---:|
| p17 | 0.6372 | 0.0901 |
| p19 | 0.7014 | 0.0422 |

So the final Gate Q null is not an event-recovery null.

## Frozen result

Final verdict: **`NO_QUERY_WINDOW_ADVANTAGE`**.

Normalized late-window geometry:

| subject | Delta effective rank (TS-PB) | p | Delta energy entropy | p |
|---|---:|---:|---:|---:|
| p17 | -31.9842 | 1.0000 | -0.062596 | 1.0000 |
| p19 | +3.5005 | 0.1077 | +0.012052 | 0.0462 |

The gate required positive, permutation-superior rank/entropy behavior in **both** participants. It did not occur.

p17 is especially destructive to the simple expansion story: TS had *lower* response dimensionality than PB, and the same direction was already present in the pre-stimulus pseudo-window (normalized Delta rank -48.1591; Delta entropy -0.078549). That looks more like a condition/state difference than a future-only expansion produced by the pulse.

p19 is interesting but insufficient: raw late rank and entropy were positive/significant, but after pooled feature normalization effective-rank evidence weakened to p=.1077 while entropy remained p=.0462. The preregistered family therefore fails. Its pre-stimulus geometry difference was near zero.

Late-response linear condition separability was 0.623 for p17 and 0.998 for p19. This says the conditions can be different without saying that one condition has more control degrees of freedom.

## What survived conceptually

A narrower statement survives because it was already demonstrated by the external experiment, not by our Gate Q metric:

> endogenous timing can change the effect of a matched intervention.

What Gate Q killed is the stronger proposed operationalization:

> theta-synchronized timing reliably **expands the dimensionality** of the future response space.

Those are different claims.

A future project may preregister a different question such as whether timing **rotates/reweights/selects** an existing response manifold rather than expanding it, but that would be a new hypothesis and must not be presented as a rescue of Gate Q.

## Current stop rule

Do not keep mining `ds006065` for a geometry metric that turns this result positive. Archive Gate Q as contact with reality.

If the broader slow-structure / fast-state / small-control-surface idea continues, its next test should come from an independently specified mechanism or task where the predicted signature is fixed *before* outcomes are inspected.
