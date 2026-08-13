"""Signal metrics for testing a cross-frequency "conductor" hypothesis.

These helpers measure statistical coordination only. They do *not* establish
that one rhythm causally controls another.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

_EPS = np.finfo(float).eps


def analytic_phase(signal: np.ndarray) -> np.ndarray:
    """Instantaneous phase along the final axis."""
    return np.angle(hilbert(np.asarray(signal), axis=-1))


def analytic_amplitude(signal: np.ndarray) -> np.ndarray:
    """Analytic amplitude envelope along the final axis."""
    return np.abs(hilbert(np.asarray(signal), axis=-1))


def _windowed_mean_real(x: np.ndarray, window_samples: int) -> np.ndarray:
    window_samples = max(1, int(window_samples))
    return uniform_filter1d(np.asarray(x, dtype=float), size=window_samples, axis=-1, mode="nearest")


def _windowed_mean_complex(z: np.ndarray, window_samples: int) -> np.ndarray:
    z = np.asarray(z)
    return _windowed_mean_real(z.real, window_samples) + 1j * _windowed_mean_real(z.imag, window_samples)


def windowed_plv(phase_a: np.ndarray, phase_b: np.ndarray, window_samples: int) -> np.ndarray:
    """Conventional 1:1 phase-locking value in a moving window.

    Use this only when a 1:1 phase relationship is meaningful. It is *not* a
    generic cross-frequency coupling metric.
    """
    phase_delta = np.asarray(phase_a) - np.asarray(phase_b)
    return np.abs(_windowed_mean_complex(np.exp(1j * phase_delta), window_samples))


def windowed_pac(
    conductor_phase: np.ndarray,
    orchestra_amplitude: np.ndarray,
    window_samples: int,
) -> np.ndarray:
    """Moving phase-amplitude coupling via normalized weighted vector strength."""
    phase = np.asarray(conductor_phase)
    amp = np.asarray(orchestra_amplitude, dtype=float)
    if phase.shape != amp.shape:
        raise ValueError(f"phase/amplitude shapes differ: {phase.shape} vs {amp.shape}")
    if np.any(amp < 0):
        raise ValueError("orchestra_amplitude must be non-negative")

    weighted = amp * np.exp(1j * phase)
    numerator = np.abs(_windowed_mean_complex(weighted, window_samples))
    denominator = _windowed_mean_real(amp, window_samples) + _EPS
    return np.clip(numerator / denominator, 0.0, 1.0)


def shifted_pac_excess(
    conductor_phase: np.ndarray,
    orchestra_amplitude: np.ndarray,
    window_samples: int,
    shift_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Actual PAC, circular-shift surrogate PAC, and their difference.

    This is a cheap *single-surrogate* screen, not a significance test. For a
    research result, use many prespecified shifts/permutations and summarize the
    resulting null distribution.
    """
    actual = windowed_pac(conductor_phase, orchestra_amplitude, window_samples)
    shifted_amp = np.roll(orchestra_amplitude, int(shift_samples), axis=-1)
    surrogate = windowed_pac(conductor_phase, shifted_amp, window_samples)
    return actual, surrogate, actual - surrogate


def old_instantaneous_plv(phase_a: np.ndarray, phase_b: np.ndarray) -> np.ndarray:
    """Historical metric retained only to demonstrate its degeneracy."""
    return np.abs(np.exp(1j * (np.asarray(phase_a) - np.asarray(phase_b))))
