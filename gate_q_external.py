"""Gate Q: external theta query-window test on OpenNeuro ds006065.

Preregistered target: p17 and p19, each with theta-synchronized (TS, task-cl)
and phase-blind (PB, task-clcontrol) stimulation.

This script intentionally uses only simple time-domain response geometry.  It
is not a claim that theta is a conscious command signal.  It asks whether a
matched intervention delivered in a different endogenous timing state reaches
a different/richer set of future network trajectories.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt, find_peaks

SEED = 20260813
N_PERM = 64
TARGET_HZ = 100.0
BASELINE = (-0.050, -0.010)
WINDOWS = {
    "late": (0.050, 0.250),
    "early": (0.015, 0.050),
    "pre": (-0.250, -0.050),
}

# Exact pulse counts and stimulation/phase contacts from the authors' released
# get_condition_info.m and get_stim_info.m.  These were frozen before looking
# at trial-level response outcomes.
SPEC = {
    "p17": {
        "anode": "D7", "cathode": "D8", "phase": "D1",
        "TS": {"task": "cl", "n": 1007},
        "PB": {"task": "clcontrol", "n": 1142},
    },
    "p19": {
        "anode": "D9", "cathode": "D10", "phase": "D1",
        "TS": {"task": "cl", "n": 523},
        "PB": {"task": "clcontrol", "n": 589},
    },
}


def _norm_name(s: str) -> str:
    return s.strip().replace(" ", "")


def _pick_name(raw: mne.io.BaseRaw, wanted: str) -> str:
    lut = {_norm_name(x).upper(): x for x in raw.ch_names}
    key = _norm_name(wanted).upper()
    if key not in lut:
        raise RuntimeError(f"Required channel {wanted!r} missing; first channels={raw.ch_names[:20]}")
    return lut[key]


def detect_events(raw: mne.io.BaseRaw, anode: str, cathode: str, expected_n: int) -> tuple[np.ndarray, dict]:
    """Recover stimulation events from the stimulation-pair artifact.

    The expected count is external metadata from the authors' released event
    index table.  We select the `expected_n` strongest bipolar derivative peaks
    subject to the published >1000 ms pulse-separation constraint.
    """
    fs = float(raw.info["sfreq"])
    a = _pick_name(raw, anode)
    c = _pick_name(raw, cathode)
    x = raw.get_data(picks=[a, c])
    bipolar = x[0] - x[1]
    score = np.abs(np.diff(bipolar, prepend=bipolar[0]))
    score[~np.isfinite(score)] = 0.0

    edge = int(round(0.30 * fs))
    peaks, _ = find_peaks(score, distance=max(1, int(math.floor(fs * 1.0))))
    peaks = peaks[(peaks >= edge) & (peaks < raw.n_times - edge)]
    if len(peaks) < expected_n:
        raise RuntimeError(f"Only {len(peaks)} >=1-s-separated peaks for expected {expected_n}")

    order = np.argsort(score[peaks])[-expected_n:]
    events = np.sort(peaks[order].astype(int))
    intervals = np.diff(events) / fs
    selected = score[events]
    unselected = np.delete(score[peaks], order) if len(peaks) > expected_n else np.array([])
    diag = {
        "expected_n": int(expected_n),
        "candidate_n": int(len(peaks)),
        "selected_n": int(len(events)),
        "min_interval_s": float(intervals.min()) if len(intervals) else np.nan,
        "median_interval_s": float(np.median(intervals)) if len(intervals) else np.nan,
        "selected_score_q10": float(np.quantile(selected, 0.10)),
        "selected_score_median": float(np.median(selected)),
        "selected_score_q90": float(np.quantile(selected, 0.90)),
        "best_unselected_score": float(unselected.max()) if len(unselected) else np.nan,
    }
    if len(intervals) and intervals.min() < 0.995:
        raise RuntimeError(f"Recovered events violate >1 s spacing: min={intervals.min():.4f}s")
    return events, diag


def phase_consistency_pre(raw: mne.io.BaseRaw, events: np.ndarray, phase_name: str) -> dict:
    """Artifact-safe phase-consistency checksum from pre-stimulus data only.

    Each trial uses -1.0 to -0.01 s, is 4-10 Hz band-passed independently, and
    phase is sampled at -0.05 s.  This is not the authors' exact online phase
    estimator; it is only an event-recovery checksum that avoids pulse leakage.
    """
    fs = float(raw.info["sfreq"])
    ch = _pick_name(raw, phase_name)
    sos = butter(4, [4.0, 10.0], btype="bandpass", fs=fs, output="sos")
    vals = []
    pre = int(round(1.0 * fs))
    stop_gap = int(round(0.01 * fs))
    target_from_event = int(round(0.05 * fs))
    for e in events:
        start = e - pre
        stop = e - stop_gap
        if start < 0 or stop <= start:
            continue
        x = raw.get_data(picks=[ch], start=start, stop=stop)[0]
        if len(x) < max(50, int(fs * 0.5)) or not np.all(np.isfinite(x)):
            continue
        try:
            xf = sosfiltfilt(sos, x)
        except ValueError:
            continue
        ph = np.angle(hilbert(xf))
        target = len(ph) - max(1, target_from_event - stop_gap)
        target = min(max(target, 0), len(ph) - 1)
        vals.append(ph[target])
    vals = np.asarray(vals)
    if len(vals) == 0:
        return {"n_phase": 0, "r_phase": np.nan, "mean_phase": np.nan}
    z = np.mean(np.exp(1j * vals))
    return {"n_phase": int(len(vals)), "r_phase": float(abs(z)), "mean_phase": float(np.angle(z))}


def _block_average(x: np.ndarray, block: int) -> np.ndarray:
    if x.shape[1] == 0:
        raise RuntimeError("Empty response window")
    chunks = []
    for i in range(0, x.shape[1], block):
        chunks.append(np.mean(x[:, i:i + block], axis=1))
    return np.stack(chunks, axis=1)


def extract_response_features(raw: mne.io.BaseRaw, events: np.ndarray, anode: str, cathode: str) -> dict[str, np.ndarray]:
    fs = float(raw.info["sfreq"])
    stim = {_pick_name(raw, anode), _pick_name(raw, cathode)}
    picks = [ch for ch in raw.ch_names if ch not in stim]
    # The OpenNeuro channel tables for these runs contain SEEG contacts only.
    # Keep exactly the same non-stimulation contacts within a run.
    rel_start, rel_stop = -0.30, 0.30
    npre = int(round(-rel_start * fs))
    npost = int(round(rel_stop * fs))
    block = max(1, int(round(fs / TARGET_HZ)))

    b0 = int(round((BASELINE[0] - rel_start) * fs))
    b1 = int(round((BASELINE[1] - rel_start) * fs))
    win_idx = {
        name: (
            int(round((lo - rel_start) * fs)),
            int(round((hi - rel_start) * fs)),
        )
        for name, (lo, hi) in WINDOWS.items()
    }

    out = {name: [] for name in WINDOWS}
    trial_rms = []
    kept_events = []
    for e in events:
        start, stop = e - npre, e + npost
        if start < 0 or stop > raw.n_times:
            continue
        seg = raw.get_data(picks=picks, start=start, stop=stop) * 1e6  # uV
        if seg.shape[1] < npre + npost or not np.all(np.isfinite(seg)):
            continue
        base = np.mean(seg[:, b0:b1], axis=1, keepdims=True)
        seg = seg - base
        for name, (i0, i1) in win_idx.items():
            v = _block_average(seg[:, i0:i1], block)
            out[name].append(v.astype(np.float32).ravel())
        late0, late1 = win_idx["late"]
        trial_rms.append(float(np.sqrt(np.mean(seg[:, late0:late1] ** 2))))
        kept_events.append(int(e))

    result = {name: np.stack(vals, axis=0) for name, vals in out.items()}
    result["trial_rms"] = np.asarray(trial_rms, dtype=float)
    result["events"] = np.asarray(kept_events, dtype=int)
    result["n_channels"] = np.asarray([len(picks)], dtype=int)
    result["channels"] = np.asarray(picks, dtype=object)
    return result


def spectral_from_kernel(K: np.ndarray) -> dict:
    row = K.mean(axis=1, keepdims=True)
    Kc = K - row - row.T + K.mean()
    eig = np.linalg.eigvalsh(Kc)
    eig = np.maximum(eig, 0.0)
    eig = eig[eig > max(1e-12, eig.max(initial=0.0) * 1e-12)]
    if len(eig) == 0:
        return {"effective_rank": 0.0, "energy_entropy": 0.0, "rank_n": 0}
    s = np.sqrt(eig)
    p = s / s.sum()
    Hs = -float(np.sum(p * np.log(np.maximum(p, 1e-300))))
    q = eig / eig.sum()
    He = -float(np.sum(q * np.log(np.maximum(q, 1e-300))))
    He_norm = He / math.log(len(q)) if len(q) > 1 else 0.0
    return {"effective_rank": float(math.exp(Hs)), "energy_entropy": He_norm, "rank_n": int(len(q))}


def kernel_for(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X @ X.T


def pooled_z(ts: np.ndarray, pb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.vstack([ts, pb]).astype(np.float64)
    mu = pooled.mean(axis=0)
    sd = pooled.std(axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    return ((ts - mu) / sd).astype(np.float32), ((pb - mu) / sd).astype(np.float32)


def balanced(ts: np.ndarray, pb: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = min(len(ts), len(pb))
    its = np.sort(rng.choice(len(ts), n, replace=False)) if len(ts) > n else np.arange(n)
    ipb = np.sort(rng.choice(len(pb), n, replace=False)) if len(pb) > n else np.arange(n)
    return ts[its], pb[ipb], its, ipb


def permutation_test(ts: np.ndarray, pb: np.ndarray, seed: int) -> dict:
    ts, pb, _, _ = balanced(ts, pb, seed)
    X = np.vstack([ts, pb]).astype(np.float64)
    n = len(ts)
    K = X @ X.T
    obs_ts = spectral_from_kernel(K[:n, :n])
    obs_pb = spectral_from_kernel(K[n:, n:])
    obs = {
        "effective_rank": obs_ts["effective_rank"] - obs_pb["effective_rank"],
        "energy_entropy": obs_ts["energy_entropy"] - obs_pb["energy_entropy"],
    }
    rng = np.random.default_rng(seed + 101)
    null = {k: [] for k in obs}
    all_idx = np.arange(2 * n)
    for _ in range(N_PERM):
        idx = rng.permutation(all_idx)
        a, b = idx[:n], idx[n:]
        ma = spectral_from_kernel(K[np.ix_(a, a)])
        mb = spectral_from_kernel(K[np.ix_(b, b)])
        for k in null:
            null[k].append(ma[k] - mb[k])
    ret = {"n_balanced": n, "obs_ts": obs_ts, "obs_pb": obs_pb, "delta": obs, "null": {}}
    for k, vals in null.items():
        vals = np.asarray(vals)
        ret["null"][k] = {
            "p_one_sided": float((1 + np.sum(vals >= obs[k])) / (len(vals) + 1)),
            "q95": float(np.quantile(vals, 0.95)),
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)),
        }
    return ret


def classifier_accuracy(ts: np.ndarray, pb: np.ndarray, seed: int) -> float:
    try:
        from sklearn.linear_model import RidgeClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except Exception:
        return float("nan")
    ts, pb, _, _ = balanced(ts, pb, seed)
    X = np.vstack([ts, pb]).astype(np.float32)
    y = np.r_[np.ones(len(ts), dtype=int), np.zeros(len(pb), dtype=int)]
    # Pooled feature z-scoring is already the intended normalization; use a
    # fixed ridge classifier rather than tuning hyperparameters to labels.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = RidgeClassifier(alpha=10.0)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=1).mean())


def process_run(data_dir: Path, subj: str, cond: str) -> dict:
    spec = SPEC[subj]
    task = spec[cond]["task"]
    vhdr = data_dir / f"sub-{subj}_task-{task}_ieeg.vhdr"
    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
    events, event_diag = detect_events(raw, spec["anode"], spec["cathode"], spec[cond]["n"])
    phase_diag = phase_consistency_pre(raw, events, spec["phase"])
    feats = extract_response_features(raw, events, spec["anode"], spec["cathode"])
    return {
        "event_diag": event_diag,
        "phase_diag": phase_diag,
        "features": feats,
        "sfreq": float(raw.info["sfreq"]),
        "n_times": int(raw.n_times),
        "duration_s": float(raw.times[-1]),
    }


def subject_gate(subj: str, runs: dict) -> dict:
    result = {"subject": subj, "windows": {}}
    tsf, pbf = runs["TS"]["features"], runs["PB"]["features"]
    result["event"] = {"TS": runs["TS"]["event_diag"], "PB": runs["PB"]["event_diag"]}
    result["phase"] = {"TS": runs["TS"]["phase_diag"], "PB": runs["PB"]["phase_diag"]}
    result["rms_uV"] = {
        "TS_median": float(np.median(tsf["trial_rms"])),
        "PB_median": float(np.median(pbf["trial_rms"])),
    }
    for wi, name in enumerate(("late", "pre", "early")):
        ts, pb, its, ipb = balanced(tsf[name], pbf[name], SEED + wi + (17 if subj == "p17" else 19))
        raw_test = permutation_test(ts, pb, SEED + 1000 + wi + (17 if subj == "p17" else 19)) if name != "early" else None
        zts, zpb = pooled_z(ts, pb)
        z_test = permutation_test(zts, zpb, SEED + 2000 + wi + (17 if subj == "p17" else 19)) if name != "early" else None
        entry = {
            "n": int(len(ts)),
            "n_features": int(ts.shape[1]),
            "raw": raw_test,
            "z": z_test,
        }
        if name == "late":
            entry["ridge_cv_accuracy"] = classifier_accuracy(zts, zpb, SEED + 3000 + (17 if subj == "p17" else 19))
        if name == "early":
            entry["raw_TS"] = spectral_from_kernel(kernel_for(ts))
            entry["raw_PB"] = spectral_from_kernel(kernel_for(pb))
            entry["z_TS"] = spectral_from_kernel(kernel_for(zts))
            entry["z_PB"] = spectral_from_kernel(kernel_for(zpb))
        result["windows"][name] = entry

    # Event recovery checksum: TS should show stronger pre-stim phase consistency
    # than PB.  This is not part of the geometry claim, but catches bad pulse recovery.
    rts = result["phase"]["TS"]["r_phase"]
    rpb = result["phase"]["PB"]["r_phase"]
    event_ok = np.isfinite(rts) and np.isfinite(rpb) and rts >= 0.30 and rts > rpb

    late = result["windows"]["late"]["z"]
    pre = result["windows"]["pre"]["z"]
    primary = (
        late["delta"]["effective_rank"] > 0
        and late["delta"]["energy_entropy"] > 0
        and late["null"]["effective_rank"]["p_one_sided"] < 0.05
        and late["null"]["energy_entropy"]["p_one_sided"] < 0.05
    )
    pre_warning = (
        abs(pre["delta"]["effective_rank"]) >= abs(late["delta"]["effective_rank"])
        and abs(pre["delta"]["energy_entropy"]) >= abs(late["delta"]["energy_entropy"])
    )
    result["event_recovery_ok"] = bool(event_ok)
    result["primary_survives"] = bool(event_ok and primary and not pre_warning)
    result["preexisting_state_warning"] = bool(pre_warning)
    return result


def write_outputs(results: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "gate_q_result.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    rows = []
    for subj, sr in results["subjects"].items():
        for win in ("late", "pre"):
            for scale in ("raw", "z"):
                t = sr["windows"][win][scale]
                rows.append({
                    "subject": subj, "window": win, "scale": scale,
                    "n": t["n_balanced"],
                    "delta_effective_rank": t["delta"]["effective_rank"],
                    "p_effective_rank": t["null"]["effective_rank"]["p_one_sided"],
                    "delta_energy_entropy": t["delta"]["energy_entropy"],
                    "p_energy_entropy": t["null"]["energy_entropy"]["p_one_sided"],
                })
    pd.DataFrame(rows).to_csv(outdir / "gate_q_metrics.csv", index=False)

    both_events = all(v["event_recovery_ok"] for v in results["subjects"].values())
    both_primary = all(v["primary_survives"] for v in results["subjects"].values())
    if not both_events:
        verdict = "EVENT_RECOVERY_FAILED"
    elif both_primary:
        verdict = "QUERY_WINDOW_ADVANTAGE_SURVIVES_PILOT"
    else:
        verdict = "NO_QUERY_WINDOW_ADVANTAGE"
    results["verdict"] = verdict
    (outdir / "gate_q_result.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    md = [
        "# Gate Q result — p17/p19 external pilot",
        "",
        f"**Verdict: `{verdict}`**",
        "",
        "This is a two-participant within-subject gate, not a population neuroscience claim.",
        "The gate was frozen before the raw response geometry was inspected.",
        "",
        "## Event recovery checksum",
        "",
        "| subject | TS pulses | PB pulses | TS pre-phase r | PB pre-phase r | recovery |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for subj, sr in results["subjects"].items():
        md.append(
            f"| {subj} | {sr['event']['TS']['selected_n']} | {sr['event']['PB']['selected_n']} | "
            f"{sr['phase']['TS']['r_phase']:.4f} | {sr['phase']['PB']['r_phase']:.4f} | {sr['event_recovery_ok']} |"
        )
    md += ["", "## Primary late-window geometry (50-250 ms)", "", "P-values are one-sided matched trial-label permutation p-values (64 permutations).", "", "| subject | scale | Delta effective rank | p | Delta energy entropy | p | ridge CV acc |", "|---|---|---:|---:|---:|---:|---:|"]
    for subj, sr in results["subjects"].items():
        for scale in ("raw", "z"):
            t = sr["windows"]["late"][scale]
            acc = sr["windows"]["late"].get("ridge_cv_accuracy", float('nan'))
            md.append(f"| {subj} | {scale} | {t['delta']['effective_rank']:.4f} | {t['null']['effective_rank']['p_one_sided']:.4f} | {t['delta']['energy_entropy']:.6f} | {t['null']['energy_entropy']['p_one_sided']:.4f} | {acc:.4f} |")
    md += ["", "## Negative-lag diagnostic (-250 to -50 ms)", "", "| subject | scale | Delta effective rank | p | Delta energy entropy | p |", "|---|---|---:|---:|---:|---:|"]
    for subj, sr in results["subjects"].items():
        for scale in ("raw", "z"):
            t = sr["windows"]["pre"][scale]
            md.append(f"| {subj} | {scale} | {t['delta']['effective_rank']:.4f} | {t['null']['effective_rank']['p_one_sided']:.4f} | {t['delta']['energy_entropy']:.6f} | {t['null']['energy_entropy']['p_one_sided']:.4f} |")
    md += ["", "## Amplitude receipt", "", "| subject | median late RMS TS (uV) | median late RMS PB (uV) |", "|---|---:|---:|"]
    for subj, sr in results["subjects"].items():
        md.append(f"| {subj} | {sr['rms_uV']['TS_median']:.3f} | {sr['rms_uV']['PB_median']:.3f} |")
    md += [
        "",
        "## Interpretation rule",
        "",
        "The interesting branch requires the geometry advantage to remain positive and permutation-superior in **both** participants after pooled per-feature z-normalization, with no equally strong pre-stimulus geometry difference. Otherwise the frozen result is `NO_QUERY_WINDOW_ADVANTAGE`.",
        "",
        "A positive result would mean only that timing relative to endogenous theta changed reachable post-stimulus response geometry under this experiment. It would not establish a universal conductor, consciousness mechanism, or causal role for PAC.",
    ]
    Path("docs/GATE_Q_RESULT.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    data_dir = Path("gate_q_data")
    results = {"dataset": "OpenNeuro ds006065", "seed": SEED, "n_perm": N_PERM, "subjects": {}}
    for subj in ("p17", "p19"):
        runs = {}
        for cond in ("TS", "PB"):
            print(f"=== {subj} {cond} ===", flush=True)
            runs[cond] = process_run(data_dir, subj, cond)
            print("event", runs[cond]["event_diag"], flush=True)
            print("phase", runs[cond]["phase_diag"], flush=True)
        results["subjects"][subj] = subject_gate(subj, runs)
        print(json.dumps(results["subjects"][subj], indent=2, default=str)[:8000], flush=True)
    write_outputs(results, Path("results"))
    print("FINAL", results.get("verdict"), flush=True)


if __name__ == "__main__":
    main()
