"""Batch runner for the RegionalAttractorExplorer pilot-field audit.

Runs the *same frozen passive screen* used by ``pilot_field_explorer.py`` across
many EEG files, writes one combined JSON receipt plus a compact CSV table, and
records annotations that MNE reads from EDF+ files.

This is intentionally not an event-conditioned analysis.  Event labels are
preserved in the receipt so that a later, separately specified gate can compare
rest/task epochs without silently changing the current P1 protocol.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from pilot_field_explorer import _clean_and_position, _load_raw, _xy_and_picks
from pilot_field_metrics import analyze_pilot_field


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
    # Stable de-duplication while retaining order.
    unique: list[Path] = []
    seen: set[str] = set()
    for p in found:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _annotation_summary(raw) -> dict:
    descriptions = [str(x) for x in raw.annotations.description]
    counts = Counter(descriptions)
    onsets = [float(x) for x in raw.annotations.onset]
    durations = [float(x) for x in raw.annotations.duration]
    return {
        "n_annotations": len(descriptions),
        "annotation_counts": dict(sorted(counts.items())),
        "annotations": [
            {"onset_s": onset, "duration_s": duration, "description": desc}
            for onset, duration, desc in zip(onsets, durations, descriptions)
        ],
    }


def _run_number(path: Path) -> int | None:
    match = re.search(r"R(\d{2})(?:\.edf)?$", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _analyze_one(path: Path, args: argparse.Namespace) -> dict:
    raw = _clean_and_position(_load_raw(path))
    full_duration = float(raw.times[-1])

    tmax = min(full_duration, args.start + args.duration)
    if tmax <= args.start:
        raise ValueError("requested time window is outside the recording")
    raw.crop(tmin=args.start, tmax=tmax)

    annotation_receipt = _annotation_summary(raw)
    xy, picks = _xy_and_picks(raw)
    data = raw.get_data(picks=picks)
    sfreq = float(raw.info["sfreq"])

    result = analyze_pilot_field(
        data,
        xy,
        sfreq,
        phase_band=tuple(args.phase_band),
        fast_band=tuple(args.fast_band),
        lag_ms=args.lag_ms,
        n_surrogates=args.surrogates,
        seed=args.seed,
    )
    payload = result.to_dict()
    payload.update(
        {
            "file": str(path),
            "file_name": path.name,
            "run_number": _run_number(path),
            "analysis_start_s": float(args.start),
            "analysis_duration_requested_s": float(args.duration),
            "analysis_duration_actual_s": float(tmax - args.start),
            "recording_duration_s": full_duration,
            "channels": [raw.ch_names[i] for i in picks],
            "companion_event_file": str(Path(str(path) + ".event"))
            if Path(str(path) + ".event").exists()
            else None,
            **annotation_receipt,
            "warning": (
                "Sensor-space traveling-wave geometry is a screening measurement. "
                "Annotations are recorded but are NOT used to select or optimize epochs in this batch. "
                "Volume conduction, reference choice, source mixing, filtering, waveform shape, "
                "EMG and other artifacts can create apparent phase gradients or cross-frequency structure."
            ),
        }
    )
    return payload


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "file_name",
        "run_number",
        "n_channels",
        "n_samples",
        "sfreq",
        "phase_fit_coherence_mean",
        "guidance_alignment",
        "guidance_n",
        "guidance_null_p",
        "guidance_null_z",
        "guidance_predictive_gain",
        "writeback_predictive_gain",
        "n_annotations",
        "annotation_counts",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), sort_keys=True)
                    if key == "annotation_counts"
                    else row.get(key)
                    for key in fields
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the frozen pilot-field screen over many EEG files and collect one receipt."
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help='EDF file(s), directory/directories, or glob(s), e.g. "S016R*.edf"',
    )
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--phase-band", nargs=2, type=float, default=(8.0, 12.0), metavar=("LOW", "HIGH"))
    ap.add_argument("--fast-band", nargs=2, type=float, default=(30.0, 45.0), metavar=("LOW", "HIGH"))
    ap.add_argument("--lag-ms", type=float, default=80.0)
    ap.add_argument("--surrogates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/pilot_field_batch.json"))
    args = ap.parse_args()

    files = _expand_inputs(args.inputs)
    if not files:
        raise SystemExit("No EDF files matched the supplied inputs.")

    rows: list[dict] = []
    for idx, path in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {path}")
        try:
            row = _analyze_one(path, args)
            rows.append(row)
            print(
                "  alignment={:+.5f} p={:.4f} pred_gain={:+.5f} writeback={:+.5f} annotations={}".format(
                    row["guidance_alignment"],
                    row["guidance_null_p"],
                    row["guidance_predictive_gain"],
                    row["writeback_predictive_gain"],
                    row["n_annotations"],
                )
            )
        except Exception as exc:  # keep a forensic receipt instead of losing the whole batch
            rows.append({"file": str(path), "file_name": path.name, "run_number": _run_number(path), "error": repr(exc)})
            print(f"  ERROR: {exc}")

    ok = [r for r in rows if "error" not in r]
    combined = {
        "schema": "regional-attractor-explorer/pilot-field-batch-v1",
        "settings": {
            "start_s": float(args.start),
            "duration_s": float(args.duration),
            "phase_band": list(args.phase_band),
            "fast_band": list(args.fast_band),
            "lag_ms": float(args.lag_ms),
            "surrogates": int(args.surrogates),
            "seed": int(args.seed),
            "event_conditioning": False,
        },
        "n_files_requested": len(files),
        "n_files_ok": len(ok),
        "n_files_failed": len(rows) - len(ok),
        "summary": {
            "mean_guidance_alignment": float(np.mean([r["guidance_alignment"] for r in ok])) if ok else None,
            "mean_guidance_predictive_gain": float(np.mean([r["guidance_predictive_gain"] for r in ok])) if ok else None,
            "mean_writeback_predictive_gain": float(np.mean([r["writeback_predictive_gain"] for r in ok])) if ok else None,
            "n_nominal_p_lt_0_05": int(sum(r["guidance_null_p"] < 0.05 for r in ok)),
            "note": (
                "The nominal p<0.05 count is descriptive only; this batch does not perform a group-level "
                "test or multiple-comparison correction. Event labels are preserved but ignored by the frozen P1 screen."
            ),
        },
        "files": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    csv_path = args.out.with_suffix(".csv")
    _write_csv(csv_path, rows)
    print(f"\nWrote {args.out}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
