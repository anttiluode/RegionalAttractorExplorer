# RegionalAttractorExplorer.py (v4 - Matplotlib Colormap Fix)
#
# This version fixes the "get_cmap() takes 2 positional arguments" error
# by updating the colormap creation to be compatible with modern versions
# of the Matplotlib library.

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import mne
from scipy.signal import butter, sosfiltfilt, hilbert, welch
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import matplotlib.pyplot as plt

# --- Brain Parcellation into Regions ---
EEG_REGIONS = {
    "Frontal_L": ['FP1', 'AF7', 'AF3', 'F7', 'F5', 'F3', 'F1'],
    "Frontal_R": ['FP2', 'AF8', 'AF4', 'F8', 'F6', 'F4', 'F2'],
    "Central": ['FCZ', 'FC1', 'FC2', 'CZ', 'C1', 'C2', 'CPZ', 'CP1', 'CP2', 'FZ', 'PZ'],
    "Temporal_L": ['F7', 'FT7', 'T7', 'TP7'],
    "Temporal_R": ['F8', 'FT8', 'T8', 'TP8'],
    "Parietal_L": ['FC5', 'FC3', 'C5', 'C3', 'CP5', 'CP3', 'P7', 'P5', 'P3', 'P1'],
    "Parietal_R": ['FC6', 'FC4', 'C6', 'C4', 'CP6', 'CP4', 'P8', 'P6', 'P4', 'P2'],
    "Occipital_L": ['P9', 'PO7', 'PO3', 'O1'],
    "Occipital_R": ['P10', 'PO8', 'PO4', 'O2'],
    "Midline_Occipital": ['OZ', 'POZ', 'IZ']
}

class RegionalAttractorExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Regional Attractor Explorer")
        self.root.geometry("1800x950")

        self.raw = None
        self.sfreq = None
        self.is_playing = False
        self.animation_timer = None
        self.frame_idx = 0
        self.frame_hop_ms = 50
        self.window_ms = 1000

        self.region_channel_indices = {}
        self.mne_values = {region: deque() for region in EEG_REGIONS}
        self.gfc_values = {region: deque() for region in EEG_REGIONS}
        self.trajectories = {region: deque() for region in EEG_REGIONS}

        self.smooth_points = tk.IntVar(value=8)
        self.zmode_var = tk.StringVar(value='PhaseSlipRate')
        self.selected_region = tk.StringVar(value="Central")

        self._build_styles()
        self._build_ui()
        self._update_status("Welcome! Load an EEG file to begin.")

    def _build_styles(self):
        st = ttk.Style()
        st.theme_use('clam')
        st.configure(".", background="#2c2c2c", foreground="#e0e0e0", font=("Segoe UI", 10))
        st.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#00aaff")
        st.configure("Clear.TButton", foreground="orange")

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(expand=True, fill=tk.BOTH)
        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(expand=True, fill=tk.BOTH)

        controls_pane = ttk.Frame(paned, width=320)
        paned.add(controls_pane, weight=1)
        ttk.Label(controls_pane, text="Controls", style="Header.TLabel").pack(pady=8, anchor="w")
        row = ttk.Frame(controls_pane); row.pack(fill=tk.X, pady=4)
        ttk.Button(row, text="📁 Load EEG", command=self.load_file).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.play_btn = ttk.Button(row, text="▶️ Play", command=self.toggle_playback); self.play_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        ttk.Button(row, text="✨ Clear", command=self.clear_trajectories, style="Clear.TButton").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))
        zrow = ttk.LabelFrame(controls_pane, text="Z-axis mode"); zrow.pack(fill=tk.X, pady=8)
        zcb = ttk.Combobox(zrow, textvariable=self.zmode_var, state="readonly", values=["PhaseSlipRate", "PhaseDiff", "PLV"]); zcb.pack(fill=tk.X, pady=4)
        arow = ttk.LabelFrame(controls_pane, text="Analysis"); arow.pack(fill=tk.X, pady=8)
        
        ttk.Label(arow, text="Window (ms)").pack(anchor='w')
        scale_window = ttk.Scale(arow, from_=250, to=3000, orient=tk.HORIZONTAL, command=lambda v: setattr(self, 'window_ms', int(float(v))))
        scale_window.set(self.window_ms)
        scale_window.pack(fill=tk.X)
        
        ttk.Label(arow, text="Hop (ms)").pack(anchor='w')
        scale_hop = ttk.Scale(arow, from_=10, to=200, orient=tk.HORIZONTAL, command=lambda v: setattr(self, 'frame_hop_ms', int(float(v))))
        scale_hop.set(self.frame_hop_ms)
        scale_hop.pack(fill=tk.X)

        ttk.Label(arow, text="Smoothing (points)").pack(anchor='w')
        ttk.Scale(arow, from_=1, to=50, orient=tk.HORIZONTAL, variable=self.smooth_points).pack(fill=tk.X)

        wrow = ttk.LabelFrame(controls_pane, text="GFC Band Weights"); wrow.pack(fill=tk.X, pady=8)
        self.weight_vars = {}
        bands = [("delta", 0.0), ("theta", 0.2), ("alpha", 1.0), ("beta", 0.1), ("gamma", 0.0)]
        for name, default in bands:
            var = tk.DoubleVar(value=default)
            self.weight_vars[name] = var
            ttk.Label(wrow, text=name.title()).pack(anchor='w', padx=5)
            scale = ttk.Scale(wrow, from_=0.0, to=2.0, orient=tk.HORIZONTAL, variable=var)
            scale.pack(fill=tk.X, padx=5, pady=(0, 5))

        self.status_lbl = ttk.Label(controls_pane, text="", wraplength=280); self.status_lbl.pack(side=tk.BOTTOM, fill=tk.X, pady=8)
        self.selected_region.trace_add("write", lambda *args: self._draw())

        plots_pane = ttk.Frame(paned)
        paned.add(plots_pane, weight=4)
        self._build_plots(plots_pane)
    
    def _build_plots(self, parent_frame):
        self.fig = Figure(figsize=(14, 9), dpi=100, facecolor="#1a1a1a")
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2, 3], height_ratios=[3, 1])
        self.ax_topo = self.fig.add_subplot(gs[0, 0])
        self.ax3d = self.fig.add_subplot(gs[0, 1], projection="3d")
        self.ax_x = self.fig.add_subplot(gs[1, 0])
        self.ax_y = self.fig.add_subplot(gs[1, 1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fig.canvas.mpl_connect('button_press_event', self._on_topo_click)

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=(("EEG Files", "*.edf *.bdf *.vhdr"), ("All files", "*.*")))
        if not path: return
        if self.is_playing: self.toggle_playback()
        self._update_status("Loading...")

        try:
            raw = mne.io.read_raw(path, preload=True, verbose=False)
            raw.pick('eeg', exclude='bads', verbose=False)
            
            rename_map = {name: name.strip().replace('.', '').upper() for name in raw.ch_names}
            raw.rename_channels(rename_map, verbose=False)

            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, on_missing='ignore', verbose=False)
            
            self.region_channel_indices.clear()
            for region, region_channels in EEG_REGIONS.items():
                indices = [raw.ch_names.index(ch_upper) for ch_upper in region_channels if ch_upper in raw.ch_names]
                if indices:
                    self.region_channel_indices[region] = indices

            raw.filter(1., 50., fir_design="firwin", verbose=False)
            self.raw = raw
            self.sfreq = float(raw.info["sfreq"])
            self.clear_trajectories()
            self._draw_region_map()
            self._update_status(f"Loaded {path.split('/')[-1]}. Click a region to begin.")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _draw_region_map(self):
        self.ax_topo.clear()
        if not self.raw: return
        
        mne.viz.plot_sensors(self.raw.info, kind='topomap', ch_type='eeg', axes=self.ax_topo, show=False)
        
        # <<< KEY FIX: Use modern Matplotlib colormap syntax >>>
        cmap = plt.colormaps.get_cmap('tab10')
        colors = cmap(np.linspace(0, 1, len(EEG_REGIONS)))
        
        pos = self.raw.get_montage().get_positions()['ch_pos']
        for i, (region, indices) in enumerate(self.region_channel_indices.items()):
            if not indices: continue
            region_pos = np.array([pos[self.raw.ch_names[idx]][:2] for idx in indices])
            self.ax_topo.scatter(region_pos[:, 0], region_pos[:, 1], s=40, color=colors[i], label=region, zorder=5)
        
        self.ax_topo.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', facecolor='#555', edgecolor='none', labelcolor='white')
        self.ax_topo.set_title("Interactive Region Selector", color='white')

    def _on_topo_click(self, event):
        if event.inaxes != self.ax_topo or not self.raw: return
        click_pos = np.array([event.xdata, event.ydata])
        pos = self.raw.get_montage().get_positions()['ch_pos']
        
        min_dist = float('inf')
        closest_ch = None
        for ch_name in self.raw.ch_names:
            ch_pos = pos[ch_name][:2]
            dist = np.linalg.norm(click_pos - ch_pos)
            if dist < min_dist:
                min_dist = dist
                closest_ch = ch_name
        
        if closest_ch:
            for region, channels in EEG_REGIONS.items():
                if closest_ch in channels:
                    self.selected_region.set(region)
                    self._update_status(f"Selected Region: {region}")
                    return

    def clear_trajectories(self):
        if self.is_playing: self.toggle_playback()
        for region in EEG_REGIONS:
            self.mne_values[region].clear()
            self.gfc_values[region].clear()
            self.trajectories[region].clear()
        self.frame_idx = 0
        self._draw()
        self._update_status("Cleared all regional trajectories.")

    def _compute_step(self, start_idx):
        window_samp = int(self.window_ms / 1000 * self.sfreq)
        stop = start_idx + window_samp
        if stop > len(self.raw.times): return False
        data_win, _ = self.raw[:, start_idx:stop]

        for region, indices in self.region_channel_indices.items():
            if not indices: continue
            region_data = data_win[indices, :]
            
            x_scalar = self._mne_alpha_power_scalar(region_data)
            gfc_ts = self._gfc_composite_timeseries(region_data)
            y_scalar = float(np.sqrt(np.mean(gfc_ts ** 2)))
            mne_alpha_ts = self._mne_alpha_timeseries(region_data)
            z = self._compute_z_value(mne_alpha_ts, gfc_ts)

            N = max(1, self.smooth_points.get())
            self.mne_values[region].append(x_scalar)
            self.gfc_values[region].append(y_scalar)
            x_sm = np.mean(list(self.mne_values[region])[-N:])
            y_sm = np.mean(list(self.gfc_values[region])[-N:])
            self.trajectories[region].append((x_sm, y_sm, z))
        return True

    def _draw(self):
        region = self.selected_region.get()
        if not region or not self.trajectories.get(region): return
        
        self.ax3d.cla()
        self.ax3d.set_facecolor("#101010")
        traj = self.trajectories[region]
        if len(traj) >= 2:
            traj_arr = np.array(traj)
            colors = plt.cm.viridis(np.linspace(0, 1, len(traj_arr)))
            self.ax3d.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2], lw=1.5, color="white", alpha=0.1)
            for i in range(len(traj_arr) - 1):
                self.ax3d.plot(traj_arr[i:i+2, 0], traj_arr[i:i+2, 1], traj_arr[i:i+2, 2], color=colors[i], lw=2)
            self.ax3d.scatter(traj_arr[-1, 0], traj_arr[-1, 1], traj_arr[-1, 2], color="white", s=50, ec="red", zorder=10)
        self.ax3d.set_title(f"Attractor for {region.replace('_', ' ')}", color='white')

        for ax in [self.ax_x, self.ax_y]: ax.cla(); ax.set_facecolor("#222")
        self.ax_x.plot(list(self.mne_values[region]), color="red")
        self.ax_y.plot(list(self.gfc_values[region]), color="cyan")
        self.ax_x.set_title(f"MNE Alpha Power ({region.replace('_', ' ')})", color='white')
        self.ax_y.set_title(f"GFC Composite ({region.replace('_', ' ')})", color='white')

        self.canvas.draw_idle()

    def toggle_playback(self):
        if not self.raw: return
        self.is_playing = not self.is_playing
        self.play_btn.config(text="⏸️ Pause" if self.is_playing else "▶️ Play")
        if self.is_playing: self._tick()

    def _tick(self):
        if not self.is_playing: return
        hop_samp = int(self.frame_hop_ms / 1000 * self.sfreq)
        ok = self._compute_step(self.frame_idx)
        if ok:
            self._draw()
            self.frame_idx += hop_samp
            if self.frame_idx + int(self.window_ms / 1000 * self.sfreq) >= len(self.raw.times):
                self.frame_idx = 0
            self.animation_timer = self.root.after(max(10, int(self.frame_hop_ms / 2)), self._tick)
        else:
            self.toggle_playback()
            self._update_status("End of file reached.")

    def _sos_band(self,lo,hi): return butter(4,[lo,hi],btype="bandpass",fs=self.sfreq,output="sos")
    def _gfc_composite_timeseries(self,d):
        c=np.zeros(d.shape[1]); band_ranges={"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,50)}
        for n,(l,h) in band_ranges.items():
            w=self.weight_vars[n].get()
            if w > 0:
                s=self._sos_band(l,h); bts=sosfiltfilt(s,d,axis=1).mean(axis=0); std=np.std(bts)
                if std > 0: bts /= std
                c += w * bts
        return c if np.any(c) else np.random.randn(d.shape[1])*1e-12
    def _mne_alpha_timeseries(self,d): return sosfiltfilt(self._sos_band(8,13),d,axis=1).mean(axis=0)
    def _mne_alpha_power_scalar(self,d):
        f,P=welch(d,fs=self.sfreq,axis=1,nperseg=min(512,d.shape[1])); b=(f>=8)&(f<=13)
        return float(P[:,b].mean())
    def _compute_z_value(self, mne_alpha_ts, gfc_ts):
        mode = self.zmode_var.get()
        try:
            if mode == "PhaseSlipRate":
                pdiff = np.unwrap(np.angle(hilbert(mne_alpha_ts)) - np.angle(hilbert(gfc_ts)))
                return float(np.gradient(pdiff).mean()) * self.sfreq
            elif mode == "PLV":
                 ph = np.angle(hilbert(mne_alpha_ts)) - np.angle(hilbert(gfc_ts))
                 return float(np.abs(np.mean(np.exp(1j * ph))))
            else: # PhaseDiff
                 return float(np.angle(np.exp(1j * (np.angle(hilbert(mne_alpha_ts)) - np.angle(hilbert(gfc_ts))))).mean())
        except Exception:
            return 0.0
    def _update_status(self, text): self.status_lbl.config(text=text)

if __name__ == "__main__":
    root = tk.Tk()
    app = RegionalAttractorExplorer(root)
    root.mainloop()