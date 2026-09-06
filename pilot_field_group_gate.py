"""Prespecified subject-level group comparison for the pilot-field audit.

This script is intentionally narrow. It compares two batch receipts produced by
``pilot_field_batch.py`` without reopening frequency, lag, reference, region, or
centroid choices.

The primary endpoint is ``guidance_predictive_gain`` because P1 required held-out
prediction improvement from the start. The group test is two-sided: before the
schizophrenia cohort is inspected we do not claim a justified direction.

``guidance_alignment`` is secondary. ``writeback_predictive_gain`` is reported
only descriptively because the healthy cohort already exposed a numerically
extreme negative value (h14), making a mean-based group test fragile.

A significant group difference would be a difference in this screening metric,
not evidence that schizophrenia is caused by, or is a disorder of, a "pilot
wave".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PRIMARY = "guidance_predictive_gain"
SECONDARY = "guidance_alignment"
DIAGNOSTIC = "writeback_predictive_gain"


def _load_batch(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in payload.get("files", []) if "error" not in r]
    if not rows:
        raise ValueError(f"no successful rows in {path}")
    for metric in (PRIMARY, SECONDARY, DIAGNOSTIC, "phase_fit_coherence_mean"):
        if any(metric not in r for r in rows):
            raise ValueError(f"{path} is missing {metric}")
    return rows


def _summary(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=float)
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "n_positive": int(np.sum(x > 0)),
        "n_negative": int(np.sum(x < 0)),
    }


def permutation_mean_difference(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 100_000,
    seed: int = 20260906,
) -> dict:
    """Two-sided Monte-Carlo label permutation test for a difference in means."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = np.r_[a, b]
    n_a = len(a)
    obs = float(np.mean(b) - np.mean(a))  # second group minus first group
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(pooled)
        delta = float(np.mean(perm[n_a:]) - np.mean(perm[:n_a]))
        exceed += abs(delta) >= abs(obs)
    p = (1 + exceed) / (1 + int(n_perm))
    return {
        "difference_second_minus_first": obs,
        "p_two_sided": float(p),
        "n_permutations": int(n_perm),
        "seed": int(seed),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Frozen subject-level group gate for two pilot-field batch receipts."
    )
    ap.add_argument("healthy", type=Path, help="healthy pilot_field_batch JSON")
    ap.add_argument("schizophrenia", type=Path, help="schizophrenia pilot_field_batch JSON")
    ap.add_argument("--permutations", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--out", type=Path, default=Path("results/pilot_field_group_gate.json"))
    args = ap.parse_args()

    healthy = _load_batch(args.healthy)
    schizophrenia = _load_batch(args.schizophrenia)

    def arr(rows: list[dict], key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in rows], dtype=float)

    hp, sp = arr(healthy, PRIMARY), arr(schizophrenia, PRIMARY)
    ha, sa = arr(healthy, SECONDARY), arr(schizophrenia, SECONDARY)
    hw, sw = arr(healthy, DIAGNOSTIC), arr(schizophrenia, DIAGNOSTIC)
    hc, sc = arr(healthy, "phase_fit_coherence_mean"), arr(schizophrenia, "phase_fit_coherence_mean")

    primary_test = permutation_mean_difference(hp, sp, args.permutations, args.seed)
    secondary_test = permutation_mean_difference(ha, sa, args.permutations, args.seed + 1)

    verdict = (
        "GROUP_DIFFERENCE_IN_P1_PREDICTIVE_GAIN"
        if primary_test["p_two_sided"] < 0.05
        else "NO_GROUP_DIFFERENCE_IN_P1_PREDICTIVE_GAIN"
    )

    payload = {
        "schema": "regional-attractor-explorer/pilot-field-group-gate-v1",
        "frozen_before_schizophrenia_results": True,
        "healthy_receipt": str(args.healthy),
        "schizophrenia_receipt": str(args.schizophrenia),
        "primary_endpoint": PRIMARY,
        "primary_test": "two-sided subject-label permutation test of mean difference",
        "healthy": {
            PRIMARY: _summary(hp),
            SECONDARY: _summary(ha),
            DIAGNOSTIC: _summary(hw),
            "phase_fit_coherence_mean": _summary(hc),
        },
        "schizophrenia": {
            PRIMARY: _summary(sp),
            SECONDARY: _summary(sa),
            DIAGNOSTIC: _summary(sw),
            "phase_fit_coherence_mean": _summary(sc),
        },
        "primary_group_test": primary_test,
        "secondary_alignment_test": secondary_test,
        "writeback_policy": (
            "descriptive only in this gate; healthy h14 produced a large negative outlier before the schizophrenia cohort was inspected"
        ),
        "verdict": verdict,
        "interpretation_boundary": (
            "A group difference is a difference in this sensor-space predictive screening metric only. "
            "It does not establish causal wave guidance, a disease mechanism, or a quantum-brain claim."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
