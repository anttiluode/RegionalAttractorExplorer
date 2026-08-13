# Gate Q result — p17/p19 external pilot

**Verdict: `EVENT_RECOVERY_FAILED`**

This is a two-participant within-subject gate, not a population neuroscience claim.
The gate was frozen before the raw response geometry was inspected.

## Event recovery checksum

| subject | TS pulses | PB pulses | TS pre-phase r | PB pre-phase r | recovery |
|---|---:|---:|---:|---:|---|
| p17 | 1007 | 1142 | 0.6526 | 0.1255 | True |
| p19 | 523 | 589 | 0.0756 | 0.0889 | False |

## Primary late-window geometry (50-250 ms)

P-values are one-sided matched trial-label permutation p-values (64 permutations).

| subject | scale | Delta effective rank | p | Delta energy entropy | p | ridge CV acc |
|---|---|---:|---:|---:|---:|---:|
| p17 | raw | -43.0370 | 1.0000 | -0.060583 | 1.0000 | 0.7403 |
| p17 | z | -50.5857 | 1.0000 | -0.091882 | 1.0000 | 0.7403 |
| p19 | raw | 6.2132 | 0.0154 | -0.019024 | 1.0000 | 0.9962 |
| p19 | z | 15.2327 | 0.0462 | 0.093782 | 0.0154 | 0.9962 |

## Negative-lag diagnostic (-250 to -50 ms)

| subject | scale | Delta effective rank | p | Delta energy entropy | p |
|---|---|---:|---:|---:|---:|
| p17 | raw | -38.3042 | 1.0000 | -0.052830 | 1.0000 |
| p17 | z | -61.8820 | 1.0000 | -0.097619 | 1.0000 |
| p19 | raw | -1.6979 | 0.7692 | 0.003881 | 0.2462 |
| p19 | z | -1.8582 | 0.8462 | 0.002093 | 0.2923 |

## Amplitude receipt

| subject | median late RMS TS (uV) | median late RMS PB (uV) |
|---|---:|---:|
| p17 | 140.853 | 102.788 |
| p19 | 300.910 | 421.229 |

## Interpretation rule

The interesting branch requires the geometry advantage to remain positive and permutation-superior in **both** participants after pooled per-feature z-normalization, with no equally strong pre-stimulus geometry difference. Otherwise the frozen result is `NO_QUERY_WINDOW_ADVANTAGE`.

A positive result would mean only that timing relative to endogenous theta changed reachable post-stimulus response geometry under this experiment. It would not establish a universal conductor, consciousness mechanism, or causal role for PAC.