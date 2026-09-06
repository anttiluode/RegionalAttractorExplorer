"""Metrics for the RegionalAttractorExplorer pilot-field audit.

This module intentionally avoids causal language. It asks whether the spatial
phase geometry of a slower oscillation contains predictive information about
where faster activity moves next, and whether the fast pattern contains
predictive information about how the slow phase field changes next.

The default use is a *screen* in sensor space. EEG reference choice, volume
conduction, source mixing, filtering and waveform shape can all create spatial
phase structure. A positive result therefore needs prespecified surrogate and
replication tests before biological interpretation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

_EPS = np.finfo(float).eps


def bandpass(data: np.ndarray, sfreq: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the final axis."""
    x = np.asarray(data, dtype=float)
    if not (0 < low < high < sfreq / 2):
        raise ValueError(f"invalid band ({low}, {high}) for sfreq={sfreq}")
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def analytic_phase(data: np.ndarray) -> np.ndarray:
    return np.angle(hilbert(np.asarray(data, dtype=float), axis=-1))


def analytic_amplitude(data: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(np.asarray(data, dtype=float), axis=-1))


def _wrap_phase(x: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * x))


def nearest_neighbor_edges(xy: np.ndarray, k: int = 4) -> np.ndarray:
    """Undirected k-nearest-neighbour edges for 2-D sensor coordinates."""
    pts = np.asarray(xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("xy must have shape (n_channels, 2)")
    n = len(pts)
    if n < 3:
        raise ValueError("need at least three positioned channels")
    k = max(1, min(int(k), n - 1))
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        nn = np.argsort(d2[i])[1 : k + 1]
        for j in nn:
            a, b = sorted((i, int(j)))
            pairs.add((a, b))
    return np.asarray(sorted(pairs), dtype=int)


def phase_gradient(
    phase: np.ndarray,
    xy: np.ndarray,
    edges: np.ndarray | None = None,
    k_neighbors: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a local planar spatial phase gradient at every sample.

    For each neighbouring sensor pair i->j we solve
        wrapped(phi_j - phi_i) ~= grad_phi dot (r_j - r_i)
    in least squares. Neighbour edges reduce phase-wrap ambiguity compared with
    all-to-all fitting. The returned residual coherence is a diagnostic only.

    Returns
    -------
    gradient : (n_times, 2)
        Estimated dphi/dx, dphi/dy.
    coherence : (n_times,)
        Circular residual coherence on the fitted edges, clipped to [0,1].
    """
    ph = np.asarray(phase, dtype=float)
    pts = np.asarray(xy, dtype=float)
    if ph.ndim != 2 or ph.shape[0] != len(pts):
        raise ValueError("phase must have shape (n_channels, n_times) matching xy")
    if edges is None:
        edges = nearest_neighbor_edges(pts, k_neighbors)
    edges = np.asarray(edges, dtype=int)
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) < 2:
        raise ValueError("edges must have shape (n_edges, 2)")

    i, j = edges[:, 0], edges[:, 1]
    dr = pts[j] - pts[i]
    if np.linalg.matrix_rank(dr) < 2:
        raise ValueError("sensor geometry/edges do not span two dimensions")
    dphi = _wrap_phase(ph[j] - ph[i])
    pinv = np.linalg.pinv(dr)
    grad = (pinv @ dphi).T

    predicted = dr @ grad.T
    residual = _wrap_phase(dphi - predicted)
    coherence = np.abs(np.mean(np.exp(1j * residual), axis=0))
    return grad, np.clip(coherence, 0.0, 1.0)


def phase_flow(gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert phase gradient to a unit propagation-direction proxy.

    For analytic phase convention phi(r,t) ~= omega*t - k.r, propagation is
    along -grad(phi). Magnitude is returned separately because absolute wave
    speed cannot be recovered from spatial phase alone.
    """
    g = np.asarray(gradient, dtype=float)
    mag = np.linalg.norm(g, axis=-1)
    flow = -g / np.maximum(mag[..., None], _EPS)
    flow[mag <= _EPS] = 0.0
    return flow, mag


def activity_centroid(amplitude: np.ndarray, xy: np.ndarray, sharpen: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Soft spatial centroid and concentration of fast-band analytic amplitude.

    A per-time median is subtracted before positive weights are sharpened. This
    makes the centroid follow relative hotspots rather than common-mode power.
    """
    amp = np.asarray(amplitude, dtype=float)
    pts = np.asarray(xy, dtype=float)
    if amp.ndim != 2 or amp.shape[0] != len(pts):
        raise ValueError("amplitude must have shape (n_channels, n_times) matching xy")
    if np.any(amp < 0):
        raise ValueError("amplitude must be non-negative")
    baseline = np.median(amp, axis=0, keepdims=True)
    excess = np.maximum(amp - baseline, 0.0)
    weights = excess ** float(sharpen)
    denom = np.sum(weights, axis=0)
    fallback = denom <= _EPS
    if np.any(fallback):
        weights[:, fallback] = np.maximum(amp[:, fallback], _EPS) ** float(sharpen)
        denom = np.sum(weights, axis=0)
    centroid = (weights.T @ pts) / np.maximum(denom[:, None], _EPS)
    centered = pts[:, None, :] - centroid[None, :, :]
    spread2 = np.sum(weights[:, :, None] * centered**2, axis=(0, 2)) / np.maximum(denom, _EPS)
    concentration = 1.0 / (np.sqrt(np.maximum(spread2, 0.0)) + 1e-12)
    return centroid, concentration


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.zeros(len(a), dtype=float)
    good = denom > _EPS
    out[good] = np.sum(a[good] * b[good], axis=1) / denom[good]
    return out


def guidance_alignment(flow: np.ndarray, centroid: np.ndarray, lag_samples: int, min_motion_quantile: float = 0.25) -> tuple[float, int]:
    """Mean alignment between slow-wave flow now and future fast-centroid motion."""
    f = np.asarray(flow, dtype=float)
    c = np.asarray(centroid, dtype=float)
    lag = int(lag_samples)
    if lag < 1 or lag >= len(c) // 2:
        raise ValueError("lag_samples must be >=1 and comfortably shorter than the record")
    disp = c[lag:] - c[:-lag]
    speed = np.linalg.norm(disp, axis=1)
    threshold = np.quantile(speed, float(min_motion_quantile))
    valid = (speed > threshold) & (np.linalg.norm(f[:-lag], axis=1) > 0)
    if not np.any(valid):
        return float("nan"), 0
    score = _cosine_rows(f[:-lag][valid], disp[valid])
    return float(np.mean(score)), int(np.sum(valid))


def circular_shift_alignment_null(
    flow: np.ndarray,
    centroid: np.ndarray,
    lag_samples: int,
    n_surrogates: int = 200,
    min_shift_samples: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, float, float]:
    """Circular-shift null for guidance alignment.

    The fast trajectory is held fixed while the slow-wave flow is shifted by a
    large amount. This preserves each signal's internal autocorrelation but
    breaks their original temporal pairing. If wave direction is effectively
    constant, this null correctly cannot distinguish guidance from a fixed
    directional bias; temporal variation is needed for this test.
    """
    f = np.asarray(flow, dtype=float)
    n = len(f)
    lag = int(lag_samples)
    if min_shift_samples is None:
        min_shift_samples = max(5 * lag, 1)
    min_shift_samples = int(min_shift_samples)
    allowed = np.arange(min_shift_samples, n - min_shift_samples)
    if len(allowed) < 5:
        raise ValueError("record too short for requested circular-shift null")
    rng = np.random.default_rng(seed)
    shifts = rng.choice(allowed, size=int(n_surrogates), replace=len(allowed) < n_surrogates)
    null = np.empty(len(shifts), dtype=float)
    for q, shift in enumerate(shifts):
        null[q], _ = guidance_alignment(np.roll(f, int(shift), axis=0), centroid, lag)
    actual, _ = guidance_alignment(f, centroid, lag)
    finite = np.isfinite(null)
    if not np.isfinite(actual) or not np.any(finite):
        return null, float("nan"), float("nan")
    p = (1.0 + np.sum(null[finite] >= actual)) / (1.0 + np.sum(finite))
    z = (actual - np.mean(null[finite])) / (np.std(null[finite]) + _EPS)
    return null, float(p), float(z)


def _ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
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


def _blocked_cv_mse(x: np.ndarray, y: np.ndarray, n_folds: int = 5) -> float:
    n = len(y)
    n_folds = max(2, min(int(n_folds), n // 20 if n >= 40 else 2))
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    preds, truths = [], []
    all_idx = np.arange(n)
    for k in range(n_folds):
        test = np.arange(edges[k], edges[k + 1])
        train = np.setdiff1d(all_idx, test, assume_unique=True)
        if len(test) == 0 or len(train) < x.shape[1] + 2:
            continue
        preds.append(_ridge_fit_predict(x[train], y[train], x[test]))
        truths.append(y[test])
    if not preds:
        return float("nan")
    pred = np.vstack(preds)
    truth = np.vstack(truths)
    return float(np.mean((truth - pred) ** 2))


def guidance_predictive_gain(flow: np.ndarray, centroid: np.ndarray, lag_samples: int, n_folds: int = 5) -> float:
    """Blocked-CV MSE improvement when slow-wave flow is added to fast dynamics.

    Baseline predicts future centroid displacement from current position and the
    previous displacement. Enhanced model adds current 2-D phase-flow direction.
    Positive values mean lower MSE with the slow field.
    """
    f = np.asarray(flow, dtype=float)
    c = np.asarray(centroid, dtype=float)
    lag = int(lag_samples)
    if 2 * lag >= len(c):
        raise ValueError("record too short for predictive guidance test")
    t = np.arange(lag, len(c) - lag)
    prev_disp = c[t] - c[t - lag]
    target = c[t + lag] - c[t]
    base = np.c_[c[t], prev_disp]
    enhanced = np.c_[base, f[t]]
    mse0 = _blocked_cv_mse(base, target, n_folds=n_folds)
    mse1 = _blocked_cv_mse(enhanced, target, n_folds=n_folds)
    return float((mse0 - mse1) / (mse0 + _EPS))


def writeback_predictive_gain(
    flow: np.ndarray,
    centroid: np.ndarray,
    fast_global_amplitude: np.ndarray,
    lag_samples: int,
    n_folds: int = 5,
) -> float:
    """Blocked-CV gain for fast pattern -> future change in slow-wave flow.

    Baseline predicts future flow from current flow. Enhanced model adds fast
    centroid and global fast amplitude. This is a predictive association test,
    not proof that fast activity writes the slow field.
    """
    f = np.asarray(flow, dtype=float)
    c = np.asarray(centroid, dtype=float)
    a = np.asarray(fast_global_amplitude, dtype=float).reshape(-1)
    if not (len(f) == len(c) == len(a)):
        raise ValueError("flow, centroid, amplitude lengths differ")
    lag = int(lag_samples)
    t = np.arange(0, len(f) - lag)
    target = f[t + lag]
    base = f[t]
    enhanced = np.c_[base, c[t], a[t]]
    mse0 = _blocked_cv_mse(base, target, n_folds=n_folds)
    mse1 = _blocked_cv_mse(enhanced, target, n_folds=n_folds)
    return float((mse0 - mse1) / (mse0 + _EPS))


@dataclass
class PilotFieldResult:
    n_channels: int
    n_samples: int
    sfreq: float
    phase_band: tuple[float, float]
    fast_band: tuple[float, float]
    lag_ms: float
    phase_fit_coherence_mean: float
    guidance_alignment: float
    guidance_n: int
    guidance_null_p: float
    guidance_null_z: float
    guidance_predictive_gain: float
    writeback_predictive_gain: float

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_pilot_field(
    data: np.ndarray,
    xy: np.ndarray,
    sfreq: float,
    phase_band: tuple[float, float] = (8.0, 12.0),
    fast_band: tuple[float, float] = (30.0, 45.0),
    lag_ms: float = 80.0,
    n_surrogates: int = 200,
    seed: int = 0,
) -> PilotFieldResult:
    """Run the complete passive pilot-field screen on channels x time data."""
    x = np.asarray(data, dtype=float)
    pts = np.asarray(xy, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(pts):
        raise ValueError("data must have shape (n_channels, n_samples) matching xy")
    slow = bandpass(x, sfreq, *phase_band)
    fast = bandpass(x, sfreq, *fast_band)
    phase = analytic_phase(slow)
    amp = analytic_amplitude(fast)
    grad, fit_coh = phase_gradient(phase, pts)
    flow, _ = phase_flow(grad)
    centroid, _ = activity_centroid(amp, pts)
    lag = max(1, int(round(float(lag_ms) * sfreq / 1000.0)))
    align, n_valid = guidance_alignment(flow, centroid, lag)
    _, p, z = circular_shift_alignment_null(
        flow, centroid, lag, n_surrogates=n_surrogates,
        min_shift_samples=max(int(round(2.0 * sfreq)), 5 * lag), seed=seed,
    )
    guide_gain = guidance_predictive_gain(flow, centroid, lag)
    fast_global = np.mean(amp, axis=0)
    write_gain = writeback_predictive_gain(flow, centroid, fast_global, lag)
    return PilotFieldResult(
        n_channels=x.shape[0], n_samples=x.shape[1], sfreq=float(sfreq),
        phase_band=tuple(map(float, phase_band)), fast_band=tuple(map(float, fast_band)),
        lag_ms=float(lag_ms), phase_fit_coherence_mean=float(np.mean(fit_coh)),
        guidance_alignment=float(align), guidance_n=int(n_valid),
        guidance_null_p=float(p), guidance_null_z=float(z),
        guidance_predictive_gain=float(guide_gain),
        writeback_predictive_gain=float(write_gain),
    )
