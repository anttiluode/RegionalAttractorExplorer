# TemporalGammaAttractorExplorer.py (v1 - Speech/Audio Gamma Gating Analysis)
#
# This version adapts the Regional Attractor Explorer to test the universal
# conductor frequency hypothesis in the temporal lobe using gamma rhythms for
# speech/audio processing analysis.

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import mne
from scipy.signal import butter, sosfiltfilt, hilbert, welch
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import matplotlib.pyplot as plt

# --- Brain Parcellation (Focus on Temporal/Speech Regions) ---
SPEECH_REGIONS = {
    "Temporal_L": ['F7', 'FT7', 'T7', 'TP7', 'T9'],  # Left temporal - primary language
    "Temporal_R": ['F8', 'FT8', 'T8', 'TP8', 'T10'], # Right temporal - prosody/melody
    "Frontal_Speech_L": ['F7', 'F5', 'F3', 'FC5', 'FC3'], # Broca's area region
    "Frontal_Speech_R": ['F8', 'F6', 'F4', 'FC6', 'FC4'], # Right frontal speech
    "Central_Motor": ['C3', 'C1', 'CZ', 'C2', 'C4'],    # Motor speech areas
    "Parietal_Assoc_L": ['CP5', 'CP3', 'P7', 'P5', 'P3'], # Left association
    "Parietal_Assoc_R": ['CP6', 'CP4', 'P8', 'P6', 'P4']  # Right association
}

class TemporalGammaAttractorExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("🎤 Temporal Gamma Attractor Explorer - Speech Analysis")
        self.root.geometry("1900x1000")

        self.raw = None
        self.sfreq = None
        self.is_playing = False
        self.animation_timer = None
        self.frame_idx = 0
        self.frame_hop_ms = 50
        self.window_ms = 1000

        self.region_channel_indices = {}
        self.gamma_values = {region: deque() for region in SPEECH_REGIONS}
        self.gfc_values = {region: deque() for region in SPEECH_REGIONS}
        self.trajectories = {region: deque() for region in SPEECH_REGIONS}

        self.smooth_points = tk.IntVar(value=8)
        self.zmode_var = tk.StringVar(value='PhaseSlipRate')
        self.selected_region = tk.StringVar(value="Temporal_L")
        
        # Gamma-specific parameters
        self.gamma_low = tk.DoubleVar(value=30.0)   # Gamma low bound
        self.gamma_high = tk.DoubleVar(value=100.0)  # Gamma high bound
        self.analysis_mode = tk.StringVar(value="Speech")  # Speech vs Silence

        self._build_styles()
        self._build_ui()
        self._update_status("Welcome! Load speech EEG data to analyze gamma gating.")

    def _build_styles(self):
        st = ttk.Style()
        st.theme_use('clam')
        st.configure(".", background="#1a1a2e", foreground="#e94560", font=("Segoe UI", 10))
        st.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#0f3460")
        st.configure("Speech.TButton", foreground="#e94560", background="#16213e")

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(expand=True, fill=tk.BOTH)
        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(expand=True, fill=tk.BOTH)

        controls_pane = ttk.Frame(paned, width=350)
        paned.add(controls_pane, weight=1)
        
        # Header
        ttk.Label(controls_pane, text="🎤 Speech Gamma Analysis", 
                 style="Header.TLabel").pack(pady=8, anchor="w")
        
        # File controls
        file_row = ttk.Frame(controls_pane)
        file_row.pack(fill=tk.X, pady=4)
        ttk.Button(file_row, text="📁 Load Speech EEG", 
                  command=self.load_file).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.play_btn = ttk.Button(file_row, text="▶️ Play", command=self.toggle_playback)
        self.play_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        ttk.Button(file_row, text="✨ Clear", command=self.clear_trajectories, 
                  style="Speech.TButton").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        # Gamma frequency controls with Nyquist awareness
        gamma_frame = ttk.LabelFrame(controls_pane, text="🧠 Gamma Band Settings")
        gamma_frame.pack(fill=tk.X, pady=8)
        
        # Nyquist frequency warning
        self.nyquist_label = ttk.Label(gamma_frame, text="Nyquist limit: Unknown", 
                                      foreground="orange")
        self.nyquist_label.pack(anchor='w')
        
        ttk.Label(gamma_frame, text="Gamma Low (Hz)").pack(anchor='w')
        self.gamma_low_scale = ttk.Scale(gamma_frame, from_=20, to=60, orient=tk.HORIZONTAL, 
                                        variable=self.gamma_low)
        self.gamma_low_scale.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(gamma_frame, text="Gamma High (Hz)").pack(anchor='w')
        self.gamma_high_scale = ttk.Scale(gamma_frame, from_=60, to=150, orient=tk.HORIZONTAL, 
                                         variable=self.gamma_high)
        self.gamma_high_scale.pack(fill=tk.X, pady=(0, 5))
        
        # Display current gamma range with warning
        self.gamma_range_label = ttk.Label(gamma_frame, text="Current: 30-100 Hz")
        self.gamma_range_label.pack()
        
        # Adaptive frequency button
        self.adapt_btn = ttk.Button(gamma_frame, text="🔧 Auto-Adapt to Sampling Rate", 
                                   command=self._adapt_gamma_to_nyquist)
        self.adapt_btn.pack(fill=tk.X, pady=5)
        
        self.gamma_low.trace('w', self._update_gamma_display)
        self.gamma_high.trace('w', self._update_gamma_display)

        # Analysis mode
        mode_frame = ttk.LabelFrame(controls_pane, text="🎯 Analysis Mode")
        mode_frame.pack(fill=tk.X, pady=8)
        mode_cb = ttk.Combobox(mode_frame, textvariable=self.analysis_mode, 
                              state="readonly", values=["Speech", "Silence", "Listening", "Rest"])
        mode_cb.pack(fill=tk.X, pady=4)

        # Z-axis mode
        zrow = ttk.LabelFrame(controls_pane, text="📊 Z-axis Metric")
        zrow.pack(fill=tk.X, pady=8)
        zcb = ttk.Combobox(zrow, textvariable=self.zmode_var, state="readonly", 
                          values=["PhaseSlipRate", "PhaseDiff", "PLV", "GammaCoupling"])
        zcb.pack(fill=tk.X, pady=4)

        # Temporal analysis controls
        arow = ttk.LabelFrame(controls_pane, text="⏱️ Temporal Analysis")
        arow.pack(fill=tk.X, pady=8)
        
        ttk.Label(arow, text="Window (ms)").pack(anchor='w')
        scale_window = ttk.Scale(arow, from_=250, to=3000, orient=tk.HORIZONTAL,
                               command=lambda v: setattr(self, 'window_ms', int(float(v))))
        scale_window.set(self.window_ms)
        scale_window.pack(fill=tk.X)
        
        ttk.Label(arow, text="Hop (ms)").pack(anchor='w')
        scale_hop = ttk.Scale(arow, from_=10, to=200, orient=tk.HORIZONTAL,
                            command=lambda v: setattr(self, 'frame_hop_ms', int(float(v))))
        scale_hop.set(self.frame_hop_ms)
        scale_hop.pack(fill=tk.X)

        ttk.Label(arow, text="Smoothing (points)").pack(anchor='w')
        ttk.Scale(arow, from_=1, to=50, orient=tk.HORIZONTAL, variable=self.smooth_points).pack(fill=tk.X)

        # GFC Band Weights (Multi-band composite)
        wrow = ttk.LabelFrame(controls_pane, text="🌊 Multi-Band Weights")
        wrow.pack(fill=tk.X, pady=8)
        self.weight_vars = {}
        bands = [("delta", 0.1), ("theta", 0.3), ("alpha", 0.2), ("beta", 0.4), ("gamma", 1.0)]
        for name, default in bands:
            var = tk.DoubleVar(value=default)
            self.weight_vars[name] = var
            ttk.Label(wrow, text=name.title()).pack(anchor='w', padx=5)
            scale = ttk.Scale(wrow, from_=0.0, to=2.0, orient=tk.HORIZONTAL, variable=var)
            scale.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Region selector
        region_frame = ttk.LabelFrame(controls_pane, text="🧠 Brain Region")
        region_frame.pack(fill=tk.X, pady=8)
        region_cb = ttk.Combobox(region_frame, textvariable=self.selected_region, 
                               state="readonly", values=list(SPEECH_REGIONS.keys()))
        region_cb.pack(fill=tk.X, pady=4)

        self.status_lbl = ttk.Label(controls_pane, text="", wraplength=320)
        self.status_lbl.pack(side=tk.BOTTOM, fill=tk.X, pady=8)
        
        self.selected_region.trace_add("write", lambda *args: self._draw())

        plots_pane = ttk.Frame(paned)
        paned.add(plots_pane, weight=4)
        self._build_plots(plots_pane)
    
    def _build_plots(self, parent_frame):
        self.fig = Figure(figsize=(16, 10), dpi=100, facecolor="#0f1419")
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2, 3], height_ratios=[3, 1])
        
        self.ax_topo = self.fig.add_subplot(gs[0, 0])
        self.ax3d = self.fig.add_subplot(gs[0, 1], projection="3d")
        self.ax_x = self.fig.add_subplot(gs[1, 0])
        self.ax_y = self.fig.add_subplot(gs[1, 1])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fig.canvas.mpl_connect('button_press_event', self._on_topo_click)

    def _update_gamma_display(self, *args):
        low = self.gamma_low.get()
        high = self.gamma_high.get()
        
        # Check Nyquist violation
        if hasattr(self, 'sfreq') and self.sfreq:
            nyquist = self.sfreq / 2
            if high > nyquist:
                self.gamma_range_label.config(text=f"⚠️ {low:.0f}-{high:.0f} Hz (EXCEEDS NYQUIST!)", 
                                            foreground="red")
            else:
                self.gamma_range_label.config(text=f"✅ {low:.0f}-{high:.0f} Hz", 
                                            foreground="green")
        else:
            self.gamma_range_label.config(text=f"Current: {low:.0f}-{high:.0f} Hz")

    def _adapt_gamma_to_nyquist(self):
        """Auto-adapt gamma range to sampling rate"""
        if not hasattr(self, 'sfreq') or not self.sfreq:
            self._update_status("⚠️ Load EEG data first to determine sampling rate")
            return
            
        nyquist = self.sfreq / 2
        
        if nyquist < 50:
            # Very low sampling rate - use high beta instead
            self.gamma_low.set(15)
            self.gamma_high.set(min(35, nyquist - 5))
            strategy = "High-Beta (15-35Hz)"
            self._update_status(f"🔧 Adapted to HIGH-BETA analysis: {strategy}")
        elif nyquist < 80:
            # Limited gamma range
            self.gamma_low.set(30)
            self.gamma_high.set(min(60, nyquist - 5))
            strategy = "Low-Gamma (30-60Hz)"
            self._update_status(f"🔧 Adapted to LOW-GAMMA analysis: {strategy}")
        else:
            # Full gamma range available
            self.gamma_low.set(30)
            self.gamma_high.set(100)
            strategy = "Full-Gamma (30-100Hz)"
            self._update_status(f"🔧 Adapted to FULL-GAMMA analysis: {strategy}")
        
        # Update scale limits
        self.gamma_high_scale.config(to=min(150, nyquist - 5))
        self._update_gamma_display()

    def _update_nyquist_info(self):
        """Update Nyquist frequency information"""
        if hasattr(self, 'sfreq') and self.sfreq:
            nyquist = self.sfreq / 2
            self.nyquist_label.config(text=f"Nyquist limit: {nyquist:.0f} Hz")
            
            # Color code based on gamma compatibility
            if nyquist >= 100:
                self.nyquist_label.config(foreground="green")
            elif nyquist >= 60:
                self.nyquist_label.config(foreground="orange") 
            else:
                self.nyquist_label.config(foreground="red")

    def _update_gamma_display(self, *args):
        low = self.gamma_low.get()
        high = self.gamma_high.get()
        
        # Check Nyquist violation
        if hasattr(self, 'sfreq') and self.sfreq:
            nyquist = self.sfreq / 2
            if high > nyquist:
                self.gamma_range_label.config(text=f"⚠️ {low:.0f}-{high:.0f} Hz (EXCEEDS NYQUIST!)", 
                                            foreground="red")
            else:
                self.gamma_range_label.config(text=f"✅ {low:.0f}-{high:.0f} Hz", 
                                            foreground="green")
        else:
            self.gamma_range_label.config(text=f"Current: {low:.0f}-{high:.0f} Hz")

    def _adapt_gamma_to_nyquist(self):
        """Auto-adapt gamma range to sampling rate"""
        if not hasattr(self, 'sfreq') or not self.sfreq:
            self._update_status("⚠️ Load EEG data first to determine sampling rate")
            return
            
        nyquist = self.sfreq / 2
        
        if nyquist < 50:
            # Very low sampling rate - use high beta instead
            self.gamma_low.set(15)
            self.gamma_high.set(min(35, nyquist - 5))
            strategy = "High-Beta (15-35Hz)"
            self._update_status(f"🔧 Adapted to HIGH-BETA analysis: {strategy}")
        elif nyquist < 80:
            # Limited gamma range
            self.gamma_low.set(30)
            self.gamma_high.set(min(60, nyquist - 5))
            strategy = "Low-Gamma (30-60Hz)"
            self._update_status(f"🔧 Adapted to LOW-GAMMA analysis: {strategy}")
        else:
            # Full gamma range available
            self.gamma_low.set(30)
            self.gamma_high.set(100)
            strategy = "Full-Gamma (30-100Hz)"
            self._update_status(f"🔧 Adapted to FULL-GAMMA analysis: {strategy}")
        
        # Update scale limits
        self.gamma_high_scale.config(to=min(150, nyquist - 5))
        self._update_gamma_display()

    def _update_nyquist_info(self):
        """Update Nyquist frequency information"""
        if hasattr(self, 'sfreq') and self.sfreq:
            nyquist = self.sfreq / 2
            self.nyquist_label.config(text=f"Nyquist limit: {nyquist:.0f} Hz")
            
            # Color code based on gamma compatibility
            if nyquist >= 100:
                self.nyquist_label.config(foreground="green")
            elif nyquist >= 60:
                self.nyquist_label.config(foreground="orange") 
            else:
                self.nyquist_label.config(foreground="red")

    def load_file(self):
        path = filedialog.askopenfilename(
            filetypes=(("EEG Files", "*.edf *.bdf *.vhdr *.fif"), ("All files", "*.*"))
        )
        if not path: 
            return
        if self.is_playing: 
            self.toggle_playback()
        self._update_status("Loading speech EEG data...")

        try:
            # Load with MNE, supporting multiple formats
            if path.endswith('.fif'):
                raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
            elif path.endswith('.vhdr'):
                raw = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
            else:
                raw = mne.io.read_raw(path, preload=True, verbose=False)
            
            raw.pick('eeg', exclude='bads', verbose=False)
            
            # Channel name standardization
            rename_map = {name: name.strip().replace('.', '').upper() for name in raw.ch_names}
            raw.rename_channels(rename_map, verbose=False)

            # Set montage
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, on_missing='ignore', verbose=False)
            
            # Map speech-relevant regions
            self.region_channel_indices.clear()
            for region, region_channels in SPEECH_REGIONS.items():
                indices = [raw.ch_names.index(ch_upper) for ch_upper in region_channels 
                          if ch_upper in raw.ch_names]
                if indices:
                    self.region_channel_indices[region] = indices

            # Filter for speech analysis (adaptive to sampling rate)
            nyquist = float(raw.info["sfreq"]) / 2
            max_freq = min(150, nyquist - 5)  # Stay 5Hz below Nyquist
            raw.filter(1., max_freq, fir_design="firwin", verbose=False)
            
            self.raw = raw
            self.sfreq = float(raw.info["sfreq"])
            
            # IMMEDIATELY auto-adapt gamma settings
            self._update_nyquist_info()
            self._adapt_gamma_to_nyquist()
            
            self.clear_trajectories()
            self._draw_region_map()
            self._update_status(f"✅ Loaded speech EEG: Nyquist: {self.sfreq/2:.0f}Hz, Gamma: {self.gamma_low.get():.0f}-{self.gamma_high.get():.0f}Hz")
            
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            self._update_status(f"Error loading file: {str(e)}")

    def _draw_region_map(self):
        self.ax_topo.clear()
        if not self.raw: 
            return
        
        try:
            mne.viz.plot_sensors(self.raw.info, kind='topomap', ch_type='eeg', 
                               axes=self.ax_topo, show=False)
            
            # Color speech regions distinctly
            cmap = plt.colormaps.get_cmap('Set1')
            colors = cmap(np.linspace(0, 1, len(SPEECH_REGIONS)))
            
            pos = self.raw.get_montage().get_positions()['ch_pos']
            for i, (region, indices) in enumerate(self.region_channel_indices.items()):
                if not indices: 
                    continue
                region_pos = np.array([pos[self.raw.ch_names[idx]][:2] for idx in indices])
                
                # Highlight temporal regions prominently
                size = 60 if 'Temporal' in region else 40
                alpha = 1.0 if 'Temporal' in region else 0.7
                
                self.ax_topo.scatter(region_pos[:, 0], region_pos[:, 1], 
                                   s=size, color=colors[i], label=region, 
                                   zorder=5, alpha=alpha, edgecolor='white', linewidth=1)
            
            self.ax_topo.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                              fontsize='small', facecolor='#333', 
                              edgecolor='none', labelcolor='white')
            self.ax_topo.set_title("Speech Regions Selector", color='white', fontsize=14)
            self.ax_topo.set_facecolor('#0f1419')
        except Exception as e:
            # Fallback: create a simple electrode map if sensor positions fail
            self.ax_topo.clear()
            self.ax_topo.set_title("Electrode Map (positions unavailable)", color='white')
            self.ax_topo.text(0.5, 0.5, f"Loaded {len(self.raw.ch_names)} channels\nClick regions below", 
                            ha='center', va='center', color='white', transform=self.ax_topo.transAxes)
            self.ax_topo.set_facecolor('#0f1419')

    def _on_topo_click(self, event):
        if event.inaxes != self.ax_topo or not self.raw: 
            return
        
        try:
            click_pos = np.array([event.xdata, event.ydata])
            pos = self.raw.get_montage().get_positions()['ch_pos']
            
            min_dist = float('inf')
            closest_ch = None
            for ch_name in self.raw.ch_names:
                if ch_name in pos:
                    ch_pos = pos[ch_name][:2]
                    dist = np.linalg.norm(click_pos - ch_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_ch = ch_name
            
            if closest_ch:
                for region, channels in SPEECH_REGIONS.items():
                    if closest_ch in channels:
                        self.selected_region.set(region)
                        self._update_status(f"Selected Speech Region: {region}")
                        return
        except Exception as e:
            # If click mapping fails, just show available regions
            regions = list(self.region_channel_indices.keys())
            if regions:
                current_idx = regions.index(self.selected_region.get()) if self.selected_region.get() in regions else 0
                next_idx = (current_idx + 1) % len(regions)
                self.selected_region.set(regions[next_idx])
                self._update_status(f"Cycled to Speech Region: {regions[next_idx]}")

    def clear_trajectories(self):
        if self.is_playing: 
            self.toggle_playback()
        for region in SPEECH_REGIONS:
            self.gamma_values[region].clear()
            self.gfc_values[region].clear()
            self.trajectories[region].clear()
        self.frame_idx = 0
        self._draw()
        self._update_status("Cleared all speech region trajectories.")

    def _compute_step(self, start_idx):
        window_samp = int(self.window_ms / 1000 * self.sfreq)
        stop = start_idx + window_samp
        if stop > len(self.raw.times): 
            return False
        data_win, _ = self.raw[:, start_idx:stop]

        for region, indices in self.region_channel_indices.items():
            if not indices: 
                continue
            region_data = data_win[indices, :]
            
            # GAMMA POWER instead of alpha power (the key change!)
            x_scalar = self._gamma_power_scalar(region_data)
            
            # Multi-band composite (moiré) - same brilliant approach
            gfc_ts = self._gfc_composite_timeseries(region_data)
            y_scalar = float(np.sqrt(np.mean(gfc_ts ** 2)))
            
            # Gamma phase relationships instead of alpha
            gamma_ts = self._gamma_timeseries(region_data)
            z = self._compute_z_value(gamma_ts, gfc_ts)

            # Smoothing
            N = max(1, self.smooth_points.get())
            self.gamma_values[region].append(x_scalar)
            self.gfc_values[region].append(y_scalar)
            x_sm = np.mean(list(self.gamma_values[region])[-N:])
            y_sm = np.mean(list(self.gfc_values[region])[-N:])
            self.trajectories[region].append((x_sm, y_sm, z))
        return True

    def _draw(self):
        region = self.selected_region.get()
        if not region or not self.trajectories.get(region): 
            return
        
        self.ax3d.cla()
        self.ax3d.set_facecolor("#000510")
        traj = self.trajectories[region]
        
        if len(traj) >= 2:
            traj_arr = np.array(traj)
            
            # Use speech-appropriate colors
            colors = plt.cm.plasma(np.linspace(0, 1, len(traj_arr)))
            
            # Ghost trail
            self.ax3d.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2], 
                          lw=1.5, color="cyan", alpha=0.15)
            
            # Colored trajectory segments
            for i in range(len(traj_arr) - 1):
                self.ax3d.plot(traj_arr[i:i+2, 0], traj_arr[i:i+2, 1], traj_arr[i:i+2, 2], 
                              color=colors[i], lw=2.5)
            
            # Current position marker
            self.ax3d.scatter(traj_arr[-1, 0], traj_arr[-1, 1], traj_arr[-1, 2], 
                            color="white", s=80, ec="red", lw=2, zorder=10)
        
        self.ax3d.set_title(f"🎤 Gamma Attractor: {region.replace('_', ' ')}", 
                          color='white', fontsize=12)
        self.ax3d.set_xlabel("Gamma Power", color='white')
        self.ax3d.set_ylabel("Multi-Band Composite", color='white')
        self.ax3d.set_zlabel("Phase Coupling", color='white')

        # Time series plots
        for ax in [self.ax_x, self.ax_y]: 
            ax.cla()
            ax.set_facecolor("#001122")
        
        if self.gamma_values[region]:
            self.ax_x.plot(list(self.gamma_values[region]), color="orange", linewidth=2)
        if self.gfc_values[region]:
            self.ax_y.plot(list(self.gfc_values[region]), color="cyan", linewidth=2)
        
        self.ax_x.set_title(f"Gamma Power ({region.replace('_', ' ')})", color='white')
        self.ax_y.set_title(f"Multi-Band Composite ({region.replace('_', ' ')})", color='white')
        
        self.ax_x.tick_params(colors='white')
        self.ax_y.tick_params(colors='white')

        self.canvas.draw_idle()

    def toggle_playback(self):
        if not self.raw: 
            return
        self.is_playing = not self.is_playing
        self.play_btn.config(text="⏸️ Pause" if self.is_playing else "▶️ Play")
        if self.is_playing: 
            self._tick()

    def _tick(self):
        if not self.is_playing: 
            return
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
            self._update_status("End of speech data reached.")

    # ============================================================================
    # GAMMA-SPECIFIC ANALYSIS FUNCTIONS (Key Innovation!)
    # ============================================================================
    
    def _sos_band(self, lo, hi): 
        """Create bandpass filter with automatic Nyquist handling"""
        # Ensure frequencies are within valid range
        nyquist = self.sfreq / 2
        lo = max(0.1, min(lo, nyquist - 0.1))  # Keep away from edges
        hi = max(lo + 0.1, min(hi, nyquist - 0.1))
        return butter(4, [lo, hi], btype="bandpass", fs=self.sfreq, output="sos")
    
    def _gamma_timeseries(self, d): 
        """Extract gamma band timeseries (replaces alpha)"""
        low = self.gamma_low.get()
        high = self.gamma_high.get()
        
        # Auto-clamp to valid range
        nyquist = self.sfreq / 2
        low = max(1.0, min(low, nyquist - 1))
        high = max(low + 1, min(high, nyquist - 1))
        
        try:
            return sosfiltfilt(self._sos_band(low, high), d, axis=1).mean(axis=0)
        except:
            # Fallback to safe range if filter fails
            return sosfiltfilt(self._sos_band(15, min(35, nyquist-1)), d, axis=1).mean(axis=0)
    
    def _gamma_power_scalar(self, d):
        """Calculate gamma power (replaces alpha power) - THE KEY METRIC"""
        low = self.gamma_low.get()
        high = self.gamma_high.get()
        
        # Auto-clamp to valid range
        nyquist = self.sfreq / 2
        low = max(1.0, min(low, nyquist - 1))
        high = max(low + 1, min(high, nyquist - 1))
        
        try:
            f, P = welch(d, fs=self.sfreq, axis=1, nperseg=min(512, d.shape[1]))
            b = (f >= low) & (f <= high)
            return float(P[:, b].mean()) if np.any(b) else 0.0
        except:
            # Fallback calculation
            return 1e-12
    
    def _gfc_composite_timeseries(self, d):
        """Multi-band composite - unchanged from original (brilliant approach)"""
        c = np.zeros(d.shape[1])
        nyquist = self.sfreq / 2
        
        # Adapt band ranges to sampling rate
        band_ranges = {
            "delta": (1, 4), 
            "theta": (4, 8), 
            "alpha": (8, 13), 
            "beta": (13, min(30, nyquist-1)), 
            "gamma": (min(30, nyquist-5), min(50, nyquist-1))
        }
        
        for n, (l, h) in band_ranges.items():
            if h <= l or h >= nyquist:  # Skip invalid bands
                continue
                
            w = self.weight_vars[n].get()
            if w > 0:
                try:
                    s = self._sos_band(l, h)
                    bts = sosfiltfilt(s, d, axis=1).mean(axis=0)
                    std = np.std(bts)
                    if std > 0: 
                        bts /= std
                    c += w * bts
                except:
                    continue  # Skip bands that fail
                    
        return c if np.any(c) else np.random.randn(d.shape[1]) * 1e-12
    
    def _compute_z_value(self, gamma_ts, gfc_ts):
        """Compute phase relationship metrics using gamma instead of alpha"""
        mode = self.zmode_var.get()
        try:
            if mode == "PhaseSlipRate":
                pdiff = np.unwrap(np.angle(hilbert(gamma_ts)) - np.angle(hilbert(gfc_ts)))
                return float(np.gradient(pdiff).mean()) * self.sfreq
            elif mode == "PLV":
                ph = np.angle(hilbert(gamma_ts)) - np.angle(hilbert(gfc_ts))
                return float(np.abs(np.mean(np.exp(1j * ph))))
            elif mode == "GammaCoupling":
                # New metric: gamma-specific coupling
                gamma_phase = np.angle(hilbert(gamma_ts))
                composite_amp = np.abs(hilbert(gfc_ts))
                return float(np.abs(np.mean(composite_amp * np.exp(1j * gamma_phase))))
            else:  # PhaseDiff
                return float(np.angle(np.exp(1j * (np.angle(hilbert(gamma_ts)) - 
                                                 np.angle(hilbert(gfc_ts))))).mean())
        except Exception:
            return 0.0
    
    def _update_status(self, text): 
        self.status_lbl.config(text=text)

# ================================================================================
# BATCH ANALYSIS FOR SPEECH VS SILENCE COMPARISON
# ================================================================================

class SpeechSilenceComparator:
    """Compare gamma attractors between speech and silence conditions"""
    
    def __init__(self, explorer):
        self.explorer = explorer
    
    def analyze_speech_silence_dataset(self, speech_files, silence_files, output_path):
        """
        Analyze speech vs silence files and generate comparison report
        similar to the SPIS analysis but for gamma gating
        """
        results = {
            'speech': {'gamma_power': [], 'trajectory_volumes': [], 'coupling': []},
            'silence': {'gamma_power': [], 'trajectory_volumes': [], 'coupling': []}
        }
        
        print("🎤 Analyzing Speech vs Silence Gamma Gating...")
        
        # Process speech files
        for file_path in speech_files:
            self.explorer.load_file(file_path)
            metrics = self._extract_gamma_metrics()
            for key, val in metrics.items():
                results['speech'][key].append(val)
        
        # Process silence files  
        for file_path in silence_files:
            self.explorer.load_file(file_path)
            metrics = self._extract_gamma_metrics()
            for key, val in metrics.items():
                results['silence'][key].append(val)
        
        # Generate comparison report
        self._generate_comparison_report(results, output_path)
        return results
    
    def _extract_gamma_metrics(self):
        """Extract key gamma metrics from current loaded data"""
        # Run through the entire file and collect metrics
        region = "Temporal_L"  # Focus on primary speech region
        
        gamma_powers = []
        trajectory_points = []
        coupling_values = []
        
        # Process entire file
        window_samp = int(self.explorer.window_ms / 1000 * self.explorer.sfreq)
        for start_idx in range(0, len(self.explorer.raw.times) - window_samp, 
                              int(self.explorer.frame_hop_ms / 1000 * self.explorer.sfreq)):
            
            data_win, _ = self.explorer.raw[:, start_idx:start_idx + window_samp]
            if region in self.explorer.region_channel_indices:
                indices = self.explorer.region_channel_indices[region]
                region_data = data_win[indices, :]
                
                gamma_power = self.explorer._gamma_power_scalar(region_data)
                gamma_powers.append(gamma_power)
                
                gfc_ts = self.explorer._gfc_composite_timeseries(region_data)
                gamma_ts = self.explorer._gamma_timeseries(region_data)
                coupling = self.explorer._compute_z_value(gamma_ts, gfc_ts)
                coupling_values.append(coupling)
                
                trajectory_points.append((gamma_power, np.sqrt(np.mean(gfc_ts ** 2)), coupling))
        
        # Calculate trajectory volume (3D convex hull volume)
        if len(trajectory_points) > 4:
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(trajectory_points)
                volume = hull.volume
            except:
                volume = 0.0
        else:
            volume = 0.0
        
        return {
            'gamma_power': np.mean(gamma_powers),
            'trajectory_volumes': volume,
            'coupling': np.mean(coupling_values)
        }
    
    def _generate_comparison_report(self, results, output_path):
        """Generate detailed comparison report"""
        speech_gamma = np.mean(results['speech']['gamma_power'])
        silence_gamma = np.mean(results['silence']['gamma_power'])
        speech_volume = np.mean(results['speech']['trajectory_volumes'])
        silence_volume = np.mean(results['silence']['trajectory_volumes'])
        speech_coupling = np.mean(results['speech']['coupling'])
        silence_coupling = np.mean(results['silence']['coupling'])
        
        gamma_ratio = speech_gamma / silence_gamma if silence_gamma > 0 else 0
        volume_ratio = speech_volume / silence_volume if silence_volume > 0 else 0
        
        report = f"""
# 🎤 SPEECH vs SILENCE GAMMA GATING ANALYSIS REPORT

## Key Findings:

### Gamma Power Analysis:
- **Speech Mean Gamma Power**: {speech_gamma:.2e}
- **Silence Mean Gamma Power**: {silence_gamma:.2e}
- **Ratio (Speech/Silence)**: {gamma_ratio:.2f}x

### Trajectory Volume Analysis:
- **Speech Mean Volume**: {speech_volume:.2e}
- **Silence Mean Volume**: {silence_volume:.2e}
- **Ratio (Speech/Silence)**: {volume_ratio:.2f}x

### Phase Coupling Analysis:
- **Speech Mean Coupling**: {speech_coupling:.4f}
- **Silence Mean Coupling**: {silence_coupling:.4f}
- **Difference**: {speech_coupling - silence_coupling:.4f}

## Interpretation:

### Expected Gamma Gating Pattern:
If the universal conductor frequency hypothesis is correct, we should see:

1. **SPEECH Condition (Temporal Lobe Active)**:
   - HIGH gamma power (conductor frequency active)
   - LARGE, organized trajectory volume 
   - STRONG phase coupling (lock-slip-relock cycles)

2. **SILENCE Condition (Temporal Lobe Idle)**:
   - LOW gamma power (conductor frequency suppressed)
   - SMALL, tight trajectory volume
   - WEAK phase coupling (random fluctuations)

### Analysis Results:
- **Gamma Ratio**: {gamma_ratio:.2f}x {'✅ SUPPORTS' if gamma_ratio > 1.5 else '❌ CONTRADICTS'} hypothesis
- **Volume Ratio**: {volume_ratio:.2f}x {'✅ SUPPORTS' if volume_ratio > 1.2 else '❌ CONTRADICTS'} hypothesis
- **Coupling Change**: {'✅ INCREASED' if speech_coupling > silence_coupling else '❌ DECREASED'} during speech

### Conclusion:
{'🎯 GAMMA GATING CONFIRMED! The temporal lobe shows the same conductor frequency pattern as occipital alpha gating.' if gamma_ratio > 1.5 and volume_ratio > 1.2 else '🤔 Gamma gating pattern unclear. May need different frequency range or analysis parameters.'}

---
Generated by Temporal Gamma Attractor Explorer
Universal Brain Dynamics Research Framework
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"📊 Analysis complete! Report saved to: {output_path}")
        return report

if __name__ == "__main__":
    root = tk.Tk()
    app = TemporalGammaAttractorExplorer(root)
    root.mainloop()