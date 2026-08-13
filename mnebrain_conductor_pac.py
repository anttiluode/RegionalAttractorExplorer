"""Audited conductor/orchestra branch for RegionalAttractorExplorer.

This keeps the original GUI and Standard Analysis path intact, but replaces the
historical source-space cross-frequency metric with a windowed
phase-amplitude-coupling (PAC) screen.

PAC is statistical association, not evidence that one rhythm causes another.
"""
from __future__ import annotations

import tkinter as tk
import mne
import numpy as np

from conductor_metrics import analytic_amplitude, analytic_phase, windowed_pac
from mnebrain_signalvs_composite3 import EEGSourceReconstructionApp


class AuditedConductorApp(EEGSourceReconstructionApp):
    def create_coordination_tab(self):
        conductor_frame = tk.LabelFrame(self.coordination_frame, text="Candidate Conductor Phase", font=("Arial", 10, "bold"))
        conductor_frame.pack(pady=10, padx=20, fill="x")
        self.conductor_var = tk.StringVar(value="alpha")
        for text, value in [("Delta (0.5-4 Hz)", "delta"), ("Theta (4-8 Hz)", "theta"), ("Alpha (8-12 Hz)", "alpha"), ("Beta (12-30 Hz)", "beta")]:
            tk.Radiobutton(conductor_frame, text=text, variable=self.conductor_var, value=value).pack(anchor="w", padx=10)

        orchestra_frame = tk.LabelFrame(self.coordination_frame, text="Candidate Orchestra Bands (faster selected bands are used)", font=("Arial", 10, "bold"))
        orchestra_frame.pack(pady=10, padx=20, fill="x")
        self.orchestra_vars = {band: tk.BooleanVar(value=True) for band in ("delta", "theta", "alpha", "beta", "gamma")}
        for i, band in enumerate(self.orchestra_vars):
            tk.Checkbutton(orchestra_frame, text=band.title(), variable=self.orchestra_vars[band]).grid(row=0, column=i, padx=5)

        metric_frame = tk.LabelFrame(self.coordination_frame, text="Visualization Metric", font=("Arial", 10, "bold"))
        metric_frame.pack(pady=10, padx=20, fill="x")
        options = [("X-Axis: Conductor Power", "Conductor Power"), ("Y-Axis: Orchestra Power", "Orchestra Power"), ("Z-Axis: Conductor Coupling (PAC)", "Conductor Coupling (PAC)"), ("Coordinated Power (Y*PAC)", "Coordinated Power (Y*PAC)")]
        self.coord_viz_var = tk.StringVar(value="Coordinated Power (Y*PAC)")
        for text, value in options:
            tk.Radiobutton(metric_frame, text=text, variable=self.coord_viz_var, value=value).pack(anchor="w", padx=10)

    def _analyze_coordination_in_source_space(self, stc_broadband):
        self.log_result("Analyzing audited conductor/orchestra PAC in source space...")
        conductor_name = self.conductor_var.get()
        conductor_freqs = self.freq_bands[conductor_name]
        conductor_center = 0.5 * sum(conductor_freqs)
        selected = [band for band, var in self.orchestra_vars.items() if var.get() and band != conductor_name]
        faster = [band for band in selected if 0.5 * sum(self.freq_bands[band]) > conductor_center]
        ignored = [band for band in selected if band not in faster]
        if ignored:
            self.log_result("PAC screen ignores slower/equal selected bands: " + ", ".join(ignored))
        if not faster:
            raise ValueError("Select at least one orchestra band faster than the conductor for PAC.")

        stc_conductor = stc_broadband.copy().filter(l_freq=conductor_freqs[0], h_freq=conductor_freqs[1], verbose=False)
        conductor_power = stc_conductor.data ** 2
        conductor_phase = analytic_phase(stc_conductor.data)
        sfreq = 1.0 / stc_broadband.tstep
        window_seconds = max(0.5, 5.0 / max(conductor_center, 1e-6))
        window_samples = max(3, int(round(window_seconds * sfreq)))
        self.log_result(f"PAC window: {window_seconds:.3f} s ({window_samples} samples; ~5 {conductor_name} cycles).")

        band_powers, band_pacs = [], []
        for band in faster:
            lo, hi = self.freq_bands[band]
            stc_band = stc_broadband.copy().filter(l_freq=lo, h_freq=hi, verbose=False)
            amplitude = analytic_amplitude(stc_band.data)
            band_powers.append(amplitude ** 2)
            band_pacs.append(windowed_pac(conductor_phase, amplitude, window_samples))

        orchestra_power = np.mean(np.stack(band_powers), axis=0)
        coupling = np.mean(np.stack(band_pacs), axis=0)
        coordinated_power = orchestra_power * coupling
        meta = dict(vertices=stc_broadband.vertices, tmin=stc_broadband.tmin, tstep=stc_broadband.tstep, subject=stc_broadband.subject)
        self.log_result("PAC is association only. A real result requires prespecified time-shift/phase-scramble nulls.")
        return {
            "Conductor Power": mne.SourceEstimate(conductor_power, **meta),
            "Orchestra Power": mne.SourceEstimate(orchestra_power, **meta),
            "Conductor Coupling (PAC)": mne.SourceEstimate(coupling, **meta),
            "Coordinated Power (Y*PAC)": mne.SourceEstimate(coordinated_power, **meta),
        }

    def _process_coordination_visualization(self, coord_metrics, metric_to_show):
        stc_viz = coord_metrics[metric_to_show]
        title = f"Coordination Audit: {metric_to_show}"
        if metric_to_show in {"Conductor Power", "Orchestra Power"}:
            params = {"colormap": "hot", "clim": dict(kind="percent", lims=[90, 95, 99]), "title": title}
        elif metric_to_show == "Conductor Coupling (PAC)":
            params = {"colormap": "viridis", "clim": dict(kind="value", lims=[0.05, 0.20, 0.50]), "title": title}
        else:
            params = {"colormap": "plasma", "clim": dict(kind="percent", lims=[95, 97, 99.9]), "title": title}
        return stc_viz, params


if __name__ == "__main__":
    try:
        mne.viz.set_3d_backend("pyvistaqt")
    except Exception:
        print("PyVistaQt backend not available, trying MNE fallback.")
    root = tk.Tk()
    app = AuditedConductorApp(root)
    root.mainloop()
