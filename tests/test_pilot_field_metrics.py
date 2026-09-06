import numpy as np

from pilot_field_metrics import (
    activity_centroid,
    circular_shift_alignment_null,
    guidance_alignment,
    nearest_neighbor_edges,
    phase_flow,
    phase_gradient,
)


def _grid(n=6):
    x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    return np.c_[x.ravel(), y.ravel()]


def test_phase_gradient_recovers_known_wave_direction():
    xy = _grid(6)
    t = np.linspace(0, 4, 800, endpoint=False)
    # analytic phase convention omega*t - k*x => propagation +x
    phase = 2 * np.pi * 10 * t[None, :] - 1.1 * xy[:, 0, None]
    phase = np.angle(np.exp(1j * phase))
    grad, coh = phase_gradient(phase, xy)
    flow, mag = phase_flow(grad)
    assert np.mean(flow[:, 0]) > 0.99
    assert abs(np.mean(flow[:, 1])) < 1e-6
    assert np.mean(coh) > 0.99
    assert np.mean(mag) > 0.5


def test_guidance_alignment_and_shift_null_on_following_packet():
    xy = _grid(7)
    n = 1200
    # First: a simple packet moving +x should align with +x flow.
    s = np.linspace(-0.8, 0.8, n)
    center = np.c_[s, np.zeros(n)]
    d2 = np.sum((xy[:, None, :] - center[None, :, :]) ** 2, axis=-1)
    amp = np.exp(-d2 / (2 * 0.22**2))
    centroid, _ = activity_centroid(amp, xy)
    flow = np.tile([1.0, 0.0], (n, 1))
    score, count = guidance_alignment(flow, centroid, lag_samples=20)
    assert score > 0.90
    assert count > 500

    # A constant flow cannot be invalidated by time shifting. Build a second
    # packet that repeatedly changes direction, then pair flow with true future motion.
    theta = np.linspace(0, 8 * np.pi, n)
    center2 = np.c_[0.65 * np.cos(theta), 0.65 * np.sin(theta)]
    d22 = np.sum((xy[:, None, :] - center2[None, :, :]) ** 2, axis=-1)
    amp2 = np.exp(-d22 / (2 * 0.20**2))
    centroid2, _ = activity_centroid(amp2, xy)
    disp = np.roll(centroid2, -20, axis=0) - centroid2
    norm = np.linalg.norm(disp, axis=1, keepdims=True)
    varying_flow = disp / np.maximum(norm, 1e-12)
    null, p, z = circular_shift_alignment_null(
        varying_flow, centroid2, lag_samples=20, n_surrogates=100,
        min_shift_samples=100, seed=4,
    )
    actual, _ = guidance_alignment(varying_flow, centroid2, 20)
    assert actual > np.nanmean(null) + 0.5
    assert p < 0.05
    assert z > 1.0


def test_neighbor_graph_is_undirected_unique():
    xy = _grid(4)
    edges = nearest_neighbor_edges(xy, k=3)
    assert len(edges) > 0
    assert np.all(edges[:, 0] < edges[:, 1])
    assert len({tuple(e) for e in edges}) == len(edges)
