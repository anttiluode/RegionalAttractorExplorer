"""Frozen event-conditioned pilot-field gate (P1E).

Primary question:
    Is the already-defined pilot-field relationship stronger during annotated
    TASK (T1/T2) than REST (T0) in PhysioNet EEGMMIDB task runs?

This is a new gate, not a rescue of whole-run P1. Signal bands, lag, reference,
sensor geometry and feature definitions remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from pilot_field_explorer import _clean_and_position, _load_raw, _xy_and_picks
from pilot_field_metrics import (
    activity_centroid,
    analytic_amplitude,
    analytic_phase,
    bandpass,
    phase_flow,
    phase_gradient,
)

_EPS = np.finfo(float).eps


def _expand_inputs(specs: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for spec in specs:
        p = Path(spec)
        if p.is_dir():
            matches = sorted(p.glob("*.edf"))
        else:
            matches = [Path(x) for x in sorted(glob.glob(spec))]
            if not matches and p.is_file():
                matches = [p]
        found.extend(matches)
    out: list[Path] = []
    seen: set[str] = set()
    for p in found:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _run_number(path: Path) -> int | None:
    m = re.search(r"R(\d{2})(?:\.edf)?$", path.name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.zeros(len(a), dtype=float)
    good = denom > _EPS
    out[good] = np.sum(a[good] * b[good], axis=1) / denom[good]
    return out


def _ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    xtr = (x_train - mu) / sd
    xte = (x_test - mu) / sd
    xtr = np.c_[np.ones(len(xtr)), xtr]
    xte = np.c_[np.ones(len(xte)), xte]
    reg = ridge * np.eye(xtr.shape[1])
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xtr.T @ xtr + reg, xtr.T @ y_train)
    return xte @ beta


def _leave_event_out_mse(x: np.ndarray, y: np.ndarray, block_id: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    block_id = np.asarray(block_id)
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    unique = np.unique(block_id)
    if len(unique) < 2:
        return float("nan")
    for block in unique:
        test = block_id == block
        train = ~test
        if np.sum(test) == 0 or np.sum(train) < x.shape[1] + 2:
            continue
        preds.append(_ridge_predict(x[train], y[train], x[test]))
        truths.append(y[test])
    if not preds:
        return float("nan")
    pred = np.vstack(preds)
    truth = np.vstack(truths)
    return float(np.mean((truth - pred) ** 2))


def _predictive_gain(flow: np.ndarray, centroid: np.ndarray, blocks: list[dict], condition: str, lag: int) -> tuple[float, int, int]:
    bases: list[np.ndarray] = []
    enhanced: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for block in blocks:
        if block["condition"] != condition:
            continue
        t = block["t"]
        if len(t) == 0:
            continue
        prev_disp = centroid[t] - centroid[t - lag]
        target = centroid[t + lag] - centroid[t]
        base = np.c_[centroid[t], prev_disp]
        bases.append(base)
        enhanced.append(np.c_[base, flow[t]])
        targets.append(target)
        ids.append(np.full(len(t), block["block_id"], dtype=int))
    if not bases:
        return float("nan"), 0, 0
    base = np.vstack(bases)
    enh = np.vstack(enhanced)
    y = np.vstack(targets)
    block_id = np.concatenate(ids)
    mse0 = _leave_event_out_mse(base, y, block_id)
    mse1 = _leave_event_out_mse(enh, y, block_id)
    gain = (mse0 - mse1) / (mse0 + _EPS)
    return float(gain), int(len(y)), int(len(np.unique(block_id)))


def _alignment(flow: np.ndarray, centroid: np.ndarray, blocks: list[dict], condition: str, lag: int, speed_threshold: float) -> tuple[float, int, int]:
    scores: list[np.ndarray] = []
    n_blocks = 0
    for block in blocks:
        if block["condition"] != condition:
            continue
        t = block["t"]
        if len(t) == 0:
            continue
        disp = centroid[t + lag] - centroid[t]
        speed = np.linalg.norm(disp, axis=1)
        valid = (speed > speed_threshold) & (np.linalg.norm(flow[t], axis=1) > 0)
        if np.any(valid):
            scores.append(_cosine_rows(flow[t][valid], disp[valid]))
            n_blocks += 1
    if not scores:
        return float("nan"), 0, 0
    s = np.concatenate(scores)
    return float(np.mean(s)), int(len(s)), int(n_blocks)


def _event_blocks(raw, sfreq: float, n_times: int, lag: int, guard_s: float) -> list[dict]:
    guard = int(round(float(guard_s) * sfreq))
    blocks: list[dict] = []
    for block_id, (onset, duration, desc) in enumerate(
        zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)
    ):
        desc = str(desc)
        if desc not in {"T0", "T1", "T2"}:
            continue
        start0 = int(raw.time_as_index(float(onset), use_rounding=True)[0])
        stop0 = int(raw.time_as_index(float(onset + duration), use_rounding=True)[0])
        start0 = max(0, min(start0, n_times))
        stop0 = max(start0, min(stop0, n_times))

        # Require previous and future lag samples to stay inside the guarded block.
        start = start0 + guard + lag
        stop = stop0 - guard - lag
        if stop <= start:
            continue
        t = np.arange(start, stop, dtype=int)
        blocks.append(
            {
                "block_id": int(block_id),
                "description": desc,
                "condition": "REST" if desc == "T0" else "TASK",
                "onset_s": float(onset),
                "duration_s": float(duration),
                "n_valid_samples": int(len(t)),
                "t": t,
            }
        )
    return blocks


def _exact_sign_flip(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "mean": None, "p_one_sided": None}
    observed = float(np.mean(x))
    null = np.empty(2 ** len(x), dtype=float)
    for i, signs in enumerate(itertools.product((-1.0, 1.0), repeat=len(x))):
        null[i] = np.mean(x * np.asarray(signs))
    p = float(np.mean(null >= observed))
    return {
        "n": int(len(x)),
        "mean": observed,
        "p_one_sided": p,
        "null_mean": float(np.mean(null)),
        "null_q95": float(np.quantile(null, 0.95)),
    }


def _analyze_one(path: Path, phase_band: tuple[float, float], fast_band: tuple[float, float], lag_ms: float, guard_s: float) -> dict:
    raw = _clean_and_position(_load_raw(path))
    xy, picks = _xy_and_picks(raw)
    data = raw.get_data(picks=picks)
    sfreq = float(raw.info["sfreq"])
    lag = max(1, int(round(lag_ms * sfreq / 1000.0)))

    slow = bandpass(data, sfreq, *phase_band)
    slow_phase = analytic_phase(slow)
    grad, coherence = phase_gradient(slow_phase, xy)
    flow, _ = phase_flow(grad)

    fast = bandpass(data, sfreq, *fast_band)
    fast_amp = analytic_amplitude(fast)
    centroid, _ = activity_centroid(fast_amp, xy)

    blocks = _event_blocks(raw, sfreq, len(flow), lag, guard_s)
    task_blocks = sum(b["condition"] == "TASK" for b in blocks)
    rest_blocks = sum(b["condition"] == "REST" for b in blocks)
    if task_blocks < 2 or rest_blocks < 2:
        raise ValueError(f"need >=2 TASK and >=2 REST blocks; got TASK={task_blocks}, REST={rest_blocks}")

    # One frozen motion threshold per run, shared by TASK and REST.
    all_speeds: list[np.ndarray] = []
    for block in blocks:
        t = block["t"]
        if len(t):
            all_speeds.append(np.linalg.norm(centroid[t + lag] - centroid[t], axis=1))
    speed_threshold = float(np.quantile(np.concatenate(all_speeds), 0.25))

    a_task, n_task, ab_task = _alignment(flow, centroid, blocks, "TASK", lag, speed_threshold)
    a_rest, n_rest, ab_rest = _alignment(flow, centroid, blocks, "REST", lag, speed_threshold)
    g_task, gp_n_task, gb_task = _predictive_gain(flow, centroid, blocks, "TASK", lag)
    g_rest, gp_n_rest, gb_rest = _predictive_gain(flow, centroid, blocks, "REST", lag)

    return {
        "file": str(path),
        "file_name": path.name,
        "run_number": _run_number(path),
        "sfreq": sfreq,
        "n_channels": int(len(picks)),
        "n_samples": int(data.shape[1]),
        "recording_duration_s": float(raw.times[-1]),
        "n_annotations": int(len(raw.annotations)),
        "phase_fit_coherence_mean": float(np.mean(coherence)),
        "task": {
            "alignment": a_task,
            "alignment_n": n_task,
            "alignment_blocks": ab_task,
            "predictive_gain": g_task,
            "predictive_n": gp_n_task,
            "predictive_blocks": gb_task,
        },
        "rest": {
            "alignment": a_rest,
            "alignment_n": n_rest,
            "alignment_blocks": ab_rest,
            "predictive_gain": g_rest,
            "predictive_n": gp_n_rest,
            "predictive_blocks": gb_rest,
        },
        "delta_alignment": float(a_task - a_rest),
        "delta_predictive_gain": float(g_task - g_rest),
        "speed_threshold": speed_threshold,
        "usable_event_blocks": int(len(blocks)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "file_name", "run_number", "phase_fit_coherence_mean",
        "task_alignment", "rest_alignment", "delta_alignment",
        "task_predictive_gain", "rest_predictive_gain", "delta_predictive_gain",
        "task_blocks", "rest_blocks", "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            if "error" in r:
                w.writerow({"file_name": r.get("file_name"), "run_number": r.get("run_number"), "error": r["error"]})
                continue
            w.writerow({
                "file_name": r["file_name"],
                "run_number": r["run_number"],
                "phase_fit_coherence_mean": r["phase_fit_coherence_mean"],
                "task_alignment": r["task"]["alignment"],
                "rest_alignment": r["rest"]["alignment"],
                "delta_alignment": r["delta_alignment"],
                "task_predictive_gain": r["task"]["predictive_gain"],
                "rest_predictive_gain": r["rest"]["predictive_gain"],
                "delta_predictive_gain": r["delta_predictive_gain"],
                "task_blocks": r["task"]["predictive_blocks"],
                "rest_blocks": r["rest"]["predictive_blocks"],
                "error": "",
            })


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen P1E TASK-vs-REST pilot-field gate for EEGMMIDB-style annotations.")
    ap.add_argument("inputs", nargs="+", help='EDF files/directories/globs, e.g. "S016R*.edf"')
    ap.add_argument("--phase-band", nargs=2, type=float, default=(8.0, 12.0), metavar=("LOW", "HIGH"))
    ap.add_argument("--fast-band", nargs=2, type=float, default=(30.0, 45.0), metavar=("LOW", "HIGH"))
    ap.add_argument("--lag-ms", type=float, default=80.0)
    ap.add_argument("--guard-s", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("results/pilot_field_event_gate.json"))
    args = ap.parse_args()

    files = _expand_inputs(args.inputs)
    files = [p for p in files if (_run_number(p) or 0) >= 3]
    if not files:
        raise SystemExit("No task EDF files (R03+) matched.")

    rows: list[dict] = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        try:
            row = _analyze_one(path, tuple(args.phase_band), tuple(args.fast_band), args.lag_ms, args.guard_s)
            rows.append(row)
            print(
                "  d_align={:+.5f}  d_pred={:+.5f}  TASK align={:+.5f} REST align={:+.5f}".format(
                    row["delta_alignment"],
                    row["delta_predictive_gain"],
                    row["task"]["alignment"],
                    row["rest"]["alignment"],
                )
            )
        except Exception as exc:
            rows.append({"file": str(path), "file_name": path.name, "run_number": _run_number(path), "error": repr(exc)})
            print(f"  ERROR: {exc}")

    ok = [r for r in rows if "error" not in r]
    align_test = _exact_sign_flip([r["delta_alignment"] for r in ok])
    pred_test = _exact_sign_flip([r["delta_predictive_gain"] for r in ok])
    passes = (
        align_test["n"] > 0
        and pred_test["n"] > 0
        and align_test["mean"] is not None
        and pred_test["mean"] is not None
        and align_test["mean"] > 0
        and pred_test["mean"] > 0
        and align_test["p_one_sided"] < 0.05
        and pred_test["p_one_sided"] < 0.05
    )
    partial = (
        align_test["n"] > 0
        and pred_test["n"] > 0
        and (
            (align_test["mean"] > 0 and align_test["p_one_sided"] < 0.05)
            or (pred_test["mean"] > 0 and pred_test["p_one_sided"] < 0.05)
        )
    )
    verdict = "EVENT_CONDITIONED_GUIDANCE_ADVANTAGE" if passes else (
        "EVENT_CONDITIONED_PARTIAL" if partial else "NO_EVENT_CONDITIONED_GUIDANCE_ADVANTAGE"
    )

    payload = {
        "schema": "regional-attractor-explorer/pilot-field-event-gate-v1",
        "status": "frozen-before-outcome",
        "settings": {
            "phase_band": list(args.phase_band),
            "fast_band": list(args.fast_band),
            "lag_ms": float(args.lag_ms),
            "guard_s": float(args.guard_s),
            "rest_labels": ["T0"],
            "task_labels": ["T1", "T2"],
            "full_recording": True,
            "cv": "leave-one-annotation-block-out",
            "group_inference": "exact one-sided sign-flip across run-level TASK-REST deltas",
        },
        "n_files_requested": len(files),
        "n_files_ok": len(ok),
        "n_files_failed": len(rows) - len(ok),
        "alignment_delta_test": align_test,
        "predictive_delta_test": pred_test,
        "verdict": verdict,
        "interpretation_boundary": (
            "These are repeated runs from one participant, not independent participants. "
            "A pass would be within-subject repeated-run evidence only."
        ),
        "files": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.out.with_suffix(".csv"), rows)
    print("\nAlignment delta exact sign-flip:", align_test)
    print("Predictive delta exact sign-flip:", pred_test)
    print("VERDICT:", verdict)
    print("Wrote", args.out)
    print("Wrote", args.out.with_suffix(".csv"))


if __name__ == "__main__":
    main()
