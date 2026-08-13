"""Gate Q v2 event plumbing: use the dataset's concatenated 3 s epochs.

The first Gate Q run tried to rediscover pulse times from stimulation artifacts.
That was unnecessary.  The OpenNeuro BrainVision payloads are the authors'
already-epoched closed-loop trials concatenated end to end: exactly 1500 samples
per trial, with 1 s pre-stimulus and 2 s post-stimulus per the released source.

This module changes ONLY event reconstruction.  It monkey-patches the frozen
`gate_q_external` process_run and reuses all response windows, geometry metrics,
normalization, nulls, seed, and kill criteria unchanged.
"""
from __future__ import annotations

import numpy as np
import mne
from pathlib import Path

import gate_q_external as gate


def grid_events(raw: mne.io.BaseRaw, anode: str, cathode: str, expected_n: int):
    if raw.n_times % expected_n != 0:
        raise RuntimeError(
            f"Run has {raw.n_times} samples, not divisible by expected {expected_n} trials"
        )
    chunk = raw.n_times // expected_n
    if not (1495 <= chunk <= 1505):
        raise RuntimeError(f"Expected ~1500 samples/trial, found {chunk}")

    # Authors' get_subj_eeg.m: cl_pre_dur=1000 ms, cl_post_dur=2000 ms.
    # The public files contain exactly 1500 resampled samples/trial, so t=0 is
    # one third of the way through each concatenated trial: sample 500.
    event_offset = int(round(chunk / 3.0))
    events = np.arange(expected_n, dtype=int) * chunk + event_offset

    # Independent artifact receipt only; this does not choose/shift events.
    a = gate._pick_name(raw, anode)
    c = gate._pick_name(raw, cathode)
    x = raw.get_data(picks=[a, c])
    bipolar = x[0] - x[1]
    d = np.abs(np.diff(bipolar, prepend=bipolar[0]))
    local = []
    off_local = []
    radius = 6
    control_offset = int(round(0.5 * chunk))
    for start in np.arange(expected_n, dtype=int) * chunk:
        e = start + event_offset
        lo, hi = max(0, e-radius), min(raw.n_times, e+radius+1)
        local.append(float(np.max(d[lo:hi])))
        q = start + control_offset
        qlo, qhi = max(0, q-radius), min(raw.n_times, q+radius+1)
        off_local.append(float(np.max(d[qlo:qhi])))

    fs = float(raw.info["sfreq"])
    diag = {
        "method": "concatenated_epoch_grid",
        "expected_n": int(expected_n),
        "selected_n": int(len(events)),
        "raw_n_times": int(raw.n_times),
        "chunk_samples": int(chunk),
        "event_offset_samples": int(event_offset),
        "chunk_duration_s": float(chunk / fs),
        "event_offset_s": float(event_offset / fs),
        "median_artifact_derivative_at_event": float(np.median(local)),
        "median_derivative_at_half_trial_control": float(np.median(off_local)),
    }
    return events, diag


def process_run(data_dir: Path, subj: str, cond: str) -> dict:
    spec = gate.SPEC[subj]
    task = spec[cond]["task"]
    vhdr = data_dir / f"sub-{subj}_task-{task}_ieeg.vhdr"
    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
    events, event_diag = grid_events(raw, spec["anode"], spec["cathode"], spec[cond]["n"])
    phase_diag = gate.phase_consistency_pre(raw, events, spec["phase"])
    feats = gate.extract_response_features(raw, events, spec["anode"], spec["cathode"])
    return {
        "event_diag": event_diag,
        "phase_diag": phase_diag,
        "features": feats,
        "sfreq": float(raw.info["sfreq"]),
        "n_times": int(raw.n_times),
        "duration_s": float(raw.times[-1]),
    }


if __name__ == "__main__":
    gate.process_run = process_run
    gate.main()
