"""Command-line EEG screen for the RegionalAttractorExplorer pilot-field audit.

Example
-------
python pilot_field_explorer.py recording.edf --start 0 --duration 60

The output is deliberately conservative: it reports predictive geometry and
surrogate statistics. It never labels a rhythm a biological "conductor".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np

from pilot_field_metrics import analyze_pilot_field


def _load_raw(path: Path) -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    readers = {
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".fif": mne.io.read_raw_fif,
        ".set": mne.io.read_raw_eeglab,
        ".vhdr": mne.io.read_raw_brainvision,
    }
    if suffix not in readers:
        raise ValueError(f"unsupported EEG format {suffix}; use one of {sorted(readers)}")
    return readers[suffix](str(path), preload=True, verbose=False)


def _clean_and_position(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw = raw.copy().pick(picks="eeg", exclude="bads")
    cleaned = {name: name.strip().replace(".", "").upper() for name in raw.ch_names}
    raw.rename_channels(cleaned)
    # Keep existing digitization if present; fill recognizable names from 10-05.
    try:
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)
    except Exception:
        pass
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def _xy_and_picks(raw: mne.io.BaseRaw) -> tuple[np.ndarray, list[int]]:
    xy, picks = [], []
    for i, ch in enumerate(raw.info["chs"]):
        loc = np.asarray(ch["loc"][:3], dtype=float)
        if np.all(np.isfinite(loc[:2])) and np.linalg.norm(loc[:2]) > 1e-8:
            xy.append(loc[:2])
            picks.append(i)
    if len(picks) < 8:
        raise ValueError(f"only {len(picks)} EEG channels have usable 2-D positions; need >= 8")
    pts = np.asarray(xy)
    # Normalize geometry only for numerical conditioning. Direction is preserved.
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    pts = pts / max(scale, 1e-12)
    return pts, picks


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit whether a slow EEG phase field predicts future fast-activity motion.")
    ap.add_argument("eeg", type=Path)
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--duration", type=float, default=60.0, help="duration in seconds")
    ap.add_argument("--phase-band", nargs=2, type=float, default=(8.0, 12.0), metavar=("LOW", "HIGH"))
    ap.add_argument("--fast-band", nargs=2, type=float, default=(30.0, 45.0), metavar=("LOW", "HIGH"))
    ap.add_argument("--lag-ms", type=float, default=80.0)
    ap.add_argument("--surrogates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    raw = _clean_and_position(_load_raw(args.eeg))
    tmax = min(raw.times[-1], args.start + args.duration)
    if tmax <= args.start:
        raise ValueError("requested time window is outside the recording")
    raw.crop(tmin=args.start, tmax=tmax)
    xy, picks = _xy_and_picks(raw)
    data = raw.get_data(picks=picks)
    sfreq = float(raw.info["sfreq"])

    result = analyze_pilot_field(
        data, xy, sfreq,
        phase_band=tuple(args.phase_band), fast_band=tuple(args.fast_band),
        lag_ms=args.lag_ms, n_surrogates=args.surrogates, seed=args.seed,
    )
    payload = result.to_dict()
    payload["file"] = str(args.eeg)
    payload["channels"] = [raw.ch_names[i] for i in picks]
    payload["warning"] = (
        "Sensor-space traveling-wave geometry is a screening measurement. "
        "Volume conduction, reference choice, source mixing, filtering, waveform shape, "
        "EMG and other artifacts can create apparent phase gradients or cross-frequency structure."
    )

    print(json.dumps(payload, indent=2))
    out = args.out or Path("results") / f"pilot_field_{args.eeg.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
