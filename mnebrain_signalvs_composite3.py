# EEG Brain Source & Coordination Explorer
#
# This version integrates the "Universal Brain Coordination Model" to visualize
# conductor power, multi-band harmony, phase-slip dynamics, and the new
# "Coordinated Power" (Y*PLV) metric in source space.

import os
import sys
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import mne
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import threading
import time

# Suppress verbose MNE logging and Qt warnings
logging.basicConfig(level=logging.WARNING)
mne.set_log_level('WARNING')
warnings.filterwarnings('ignore', message='.*QApplication.*')
warnings.filterwarnings('ignore', message='.*QWindowsWindow.*')

class SourceReconstructor:
    """Handles different source reconstruction methods"""
    
    def __init__(self):
        self.methods = {
            'sLORETA': {'method': 'sLORETA', 'lambda2': 1.0 / 9.0},
            'dSPM': {'method': 'dSPM', 'lambda2': 1.0 / 9.0},
            'MNE': {'method': 'MNE', 'lambda2': 1.0 / 9.0},
            'eLORETA': {'method': 'eLORETA', 'lambda2': 1.0 / 9.0}
        }
    
    def reconstruct(self, raw, inverse_operator, method='sLORETA'):
        """Apply inverse solution with specified method"""
        params = self.methods[method]
        stc = mne.minimum_norm.apply_inverse_raw(
            raw, inverse_operator,
            lambda2=params['lambda2'],
            method=params['method'],
            verbose=False
        )
        return stc

class PreprocessingPipeline:
    """Handles EEG preprocessing steps"""
    
    @staticmethod
    def detect_bad_channels(raw, threshold=3.0):
        """Simple bad channel detection based on variance"""
        data = raw.get_data()
        channel_vars = np.var(data, axis=1)
        # Use median and median absolute deviation for robustness against outliers
        median_var = np.median(channel_vars)
        mad = np.median(np.abs(channel_vars - median_var))
        if mad == 0:
            return []
        z_scores = 0.6745 * np.abs(channel_vars - median_var) / mad
        bad_channels = [raw.ch_names[i] for i in np.where(z_scores > threshold)[0]]
        return bad_channels
    
    @staticmethod
    def remove_artifacts(raw, method='basic'):
        """Remove common artifacts"""
        if method == 'basic':
            # Simple high-pass filter to remove drift
            raw.filter(l_freq=0.5, h_freq=None, fir_design='firwin', verbose=False)
            
            # Notch filter for line noise - check power line frequency
            # Europe = 50 Hz, Americas = 60 Hz
            line_freq = 50  # Default to European standard
            
            # Apply notch filter for line noise and harmonics
            # Only apply to frequencies below Nyquist
            nyquist = raw.info['sfreq'] / 2.0
            freqs = []
            for harmonic in range(1, 5):  # First 4 harmonics
                freq = line_freq * harmonic
                if freq < nyquist - 1:  # Leave 1 Hz margin from Nyquist
                    freqs.append(freq)
            
            if freqs:
                raw.notch_filter(freqs, fir_design='firwin', verbose=False)
        
        elif method == 'ica':
            # ICA-based artifact removal (simplified)
            try:
                from mne.preprocessing import ICA
                ica = ICA(n_components=min(15, len(raw.ch_names)-1), random_state=42)
                ica.fit(raw, verbose=False)
                
                # Find EOG artifacts
                eog_indices, _ = ica.find_bads_eog(raw, verbose=False)
                ica.exclude = eog_indices[:2]  # Remove top 2 EOG components
                raw = ica.apply(raw, verbose=False)
            except Exception as e:
                print(f"ICA failed, falling back to basic filtering: {e}")
                # Fallback to basic if ICA fails
                PreprocessingPipeline.remove_artifacts(raw, 'basic')
        
        return raw

class EEGSourceReconstructionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Brain Source & Coordination Explorer")
        self.root.geometry("700x900")
        
        # Make window resizable
        self.root.resizable(True, True)
        
        # Initialize components
        self.reconstructor = SourceReconstructor()
        self.preprocessing = PreprocessingPipeline()
        self.processing_thread = None
        self.brain_figures = []  # Keep track of brain figures
        
        # Create UI
        self.create_ui()
        
    def create_ui(self):
        """Create the user interface"""
        # Title
        title_label = tk.Label(self.root, text="EEG Brain Source & Coordination Explorer", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Analysis type frame
        analysis_type_frame = tk.Frame(self.root)
        analysis_type_frame.pack(pady=5)
        self.analysis_mode = tk.StringVar(value="coordination") # Default to new mode
        tk.Radiobutton(analysis_type_frame, text="Standard Analysis", variable=self.analysis_mode, value="standard", command=self.switch_tabs).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(analysis_type_frame, text="Coordination Model", variable=self.analysis_mode, value="coordination", command=self.switch_tabs).pack(side=tk.LEFT, padx=10)

        # Standard Analysis tab
        self.standard_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.standard_frame, text="Standard Settings")
        self.create_standard_tab()

        # Coordination Model tab
        self.coordination_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.coordination_frame, text="Coordination Model Settings")
        self.create_coordination_tab()
        
        # Advanced tab (common to both)
        self.advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.advanced_frame, text="Advanced Settings")
        self.create_advanced_tab()
        
        # Results tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Log & Results")
        self.create_results_tab()
        
        # Action buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.load_button = tk.Button(button_frame, text="Load EEG File", 
                                     command=self.load_file,
                                     bg="#2196F3", fg="white", font=("Arial", 10))
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.process_button = tk.Button(button_frame, text="Process & Reconstruct", 
                                        command=self.run_reconstruction,
                                        bg="#4CAF50", fg="white", font=("Arial", 10),
                                        state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(button_frame, text="Stop", 
                                     command=self.stop_processing,
                                     bg="#f44336", fg="white", font=("Arial", 10),
                                     state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.close_brain_button = tk.Button(button_frame, text="Close 3D Views", 
                                            command=self.close_brain_views,
                                            bg="#FF9800", fg="white", font=("Arial", 10))
        self.close_brain_button.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(fill=tk.X, padx=20, pady=5)
        
        # Status
        self.status_label = tk.Label(self.root, text="Ready to load EEG file...", 
                                     wraplength=650)
        self.status_label.pack(pady=5)
        
        # File info
        self.file_info_label = tk.Label(self.root, text="", fg="blue", wraplength=650)
        self.file_info_label.pack(pady=5)

        self.switch_tabs()

    def switch_tabs(self):
        """Show/hide tabs based on analysis mode"""
        if self.analysis_mode.get() == "standard":
            self.notebook.add(self.standard_frame)
            self.notebook.hide(self.coordination_frame)
        else:
            self.notebook.add(self.coordination_frame)
            self.notebook.hide(self.standard_frame)

    def create_standard_tab(self):
        """Create basic settings tab"""
        # Frequency band selection
        freq_frame = tk.LabelFrame(self.standard_frame, text="Frequency Band", 
                                   font=("Arial", 10, "bold"))
        freq_frame.pack(pady=10, padx=20, fill="x")
        
        self.freq_var = tk.StringVar(value="alpha")
        freq_options = [
            ("Delta (0.5-4 Hz)", "delta"), ("Theta (4-8 Hz)", "theta"), 
            ("Alpha (8-12 Hz)", "alpha"), ("Beta (12-30 Hz)", "beta"),
            ("Gamma (30-50 Hz)", "gamma"), ("Broadband (0.5-50 Hz)", "broadband")
        ]
        
        for i, (text, value) in enumerate(freq_options):
            tk.Radiobutton(freq_frame, text=text, variable=self.freq_var, 
                           value=value, font=("Arial", 9)).grid(row=i//2, column=i%2, 
                                                               sticky="w", padx=10, pady=2)
        
        # Visualization type
        viz_frame = tk.LabelFrame(self.standard_frame, text="Visualization Type", 
                                  font=("Arial", 10, "bold"))
        viz_frame.pack(pady=10, padx=20, fill="x")
        
        self.viz_var = tk.StringVar(value="power")
        viz_options = [
            ("Power Distribution", "power"), ("Phase Patterns", "phase"),
            ("Raw Amplitude", "raw"), ("Statistical Map (z-score)", "stats")
        ]
        
        for i, (text, value) in enumerate(viz_options):
            tk.Radiobutton(viz_frame, text=text, variable=self.viz_var, 
                           value=value, font=("Arial", 9)).pack(anchor="w", padx=10)
    
    def create_coordination_tab(self):
        """Create coordination model settings tab"""
        # Conductor selection
        conductor_frame = tk.LabelFrame(self.coordination_frame, text="Conductor Frequency",
                                        font=("Arial", 10, "bold"))
        conductor_frame.pack(pady=10, padx=20, fill="x")
        
        self.conductor_var = tk.StringVar(value="alpha")
        conductor_options = [
            ("Alpha (8-12 Hz)", "alpha"), ("Gamma (30-50 Hz)", "gamma"),
            ("Beta (12-30 Hz)", "beta"), ("Theta (4-8 Hz)", "theta")
        ]
        for text, value in conductor_options:
            tk.Radiobutton(conductor_frame, text=text, variable=self.conductor_var,
                           value=value).pack(anchor="w", padx=10)

        # Orchestra (Moiré) selection
        orchestra_frame = tk.LabelFrame(self.coordination_frame, text="Orchestra (Moiré Composite) Frequencies",
                                        font=("Arial", 10, "bold"))
        orchestra_frame.pack(pady=10, padx=20, fill="x")
        self.orchestra_vars = {
            "delta": tk.BooleanVar(value=True), "theta": tk.BooleanVar(value=True),
            "alpha": tk.BooleanVar(value=True), "beta": tk.BooleanVar(value=True),
            "gamma": tk.BooleanVar(value=True)
        }
        for i, band in enumerate(self.orchestra_vars):
            tk.Checkbutton(orchestra_frame, text=f"{band.title()}",
                           variable=self.orchestra_vars[band]).grid(row=0, column=i, padx=5)

        # Visualization Metric
        metric_frame = tk.LabelFrame(self.coordination_frame, text="Visualization Metric",
                                     font=("Arial", 10, "bold"))
        metric_frame.pack(pady=10, padx=20, fill="x")

        # --- CHANGE 2: CORRECTED LOGIC --- Renamed metric for clarity
        metric_options = [
            ("X-Axis: Conductor Power", "Conductor Power"),
            ("Y-Axis: Moiré Harmony", "Moiré Harmony"),
            ("Z-Axis: Phase-Slip Rate", "Phase-Slip Rate"),
            ("Coordinated Power (Y*PLV)", "Coordinated Power (Y*PLV)")
        ]
        self.coord_viz_var = tk.StringVar()
        for text, value in metric_options:
            tk.Radiobutton(metric_frame, text=text, variable=self.coord_viz_var,
                           value=value).pack(anchor="w", padx=10)
        # Set the default choice to the new, correct metric
        self.coord_viz_var.set("Coordinated Power (Y*PLV)")


    def create_advanced_tab(self):
        """Create advanced settings tab"""
        # Preprocessing options
        preproc_frame = tk.LabelFrame(self.advanced_frame, text="Preprocessing", 
                                      font=("Arial", 10, "bold"))
        preproc_frame.pack(pady=10, padx=20, fill="x")
        
        self.remove_bad_channels_var = tk.BooleanVar(value=True)
        tk.Checkbutton(preproc_frame, text="Automatically detect and remove bad channels",
                       variable=self.remove_bad_channels_var).pack(anchor="w", padx=10, pady=2)
        
        self.artifact_removal_var = tk.StringVar(value="basic")
        tk.Label(preproc_frame, text="Artifact removal:").pack(anchor="w", padx=10, pady=2)
        tk.Radiobutton(preproc_frame, text="Basic (filters only)", 
                       variable=self.artifact_removal_var, value="basic").pack(anchor="w", padx=30)
        tk.Radiobutton(preproc_frame, text="ICA-based (removes EOG artifacts)", 
                       variable=self.artifact_removal_var, value="ica").pack(anchor="w", padx=30)
        
        # Source reconstruction options
        source_frame = tk.LabelFrame(self.advanced_frame, text="Source Reconstruction", 
                                     font=("Arial", 10, "bold"))
        source_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(source_frame, text="Inverse method:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.method_var = tk.StringVar(value="sLORETA")
        method_menu = ttk.Combobox(source_frame, textvariable=self.method_var,
                                   values=["sLORETA", "dSPM", "MNE", "eLORETA"],
                                   state="readonly", width=15)
        method_menu.grid(row=0, column=1, padx=10, pady=5)
        
        tk.Label(source_frame, text="Source space:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.spacing_var = tk.StringVar(value="ico5")
        spacing_menu = ttk.Combobox(source_frame, textvariable=self.spacing_var,
                                    values=["ico4", "ico5", "oct5", "oct6"],
                                    state="readonly", width=15)
        spacing_menu.grid(row=1, column=1, padx=10, pady=5)
        
        # Time window (moved here for common access)
        time_frame = tk.LabelFrame(self.advanced_frame, text="Time Window (seconds)",
                                   font=("Arial", 10, "bold"))
        time_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(time_frame, text="Analyze from:").grid(row=0, column=0, padx=5, pady=5)
        self.time_start_var = tk.DoubleVar(value=0.0)
        self.time_start_spin = tk.Spinbox(time_frame, from_=0, to=1000, increment=0.5,
                                          textvariable=self.time_start_var, width=10)
        self.time_start_spin.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(time_frame, text="to").grid(row=0, column=2, padx=5, pady=5)
        
        self.time_end_var = tk.DoubleVar(value=10.0)
        self.time_end_spin = tk.Spinbox(time_frame, from_=0, to=1000, increment=0.5,
                                        textvariable=self.time_end_var, width=10)
        self.time_end_spin.grid(row=0, column=3, padx=5, pady=5)

    def create_results_tab(self):
        """Create results visualization tab"""
        self.results_text = tk.Text(self.results_frame, height=10, wrap=tk.WORD, bg="#f0f0f0")
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(self.results_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)

    def load_file(self):
        """Load EEG file"""
        filepath = filedialog.askopenfilename(
            title="Select EEG File",
            filetypes=[("EEG Files", "*.edf *.bdf *.fif *.set *.vhdr"), 
                       ("All files", "*.*")]
        )
        if not filepath:
            return
        
        try:
            self.update_status("Loading EEG file...")
            self.progress['value'] = 10
            
            self.raw = mne.io.read_raw(filepath, preload=True, verbose=False)
            self.raw.pick_types(eeg=True, exclude=[])
            
            n_channels = len(self.raw.ch_names)
            duration = self.raw.times[-1]
            sfreq = self.raw.info['sfreq']
            self.filename = os.path.basename(filepath)
            
            info_text = (f"Loaded: {self.filename}\n"
                         f"Channels: {n_channels} | Duration: {duration:.1f}s | "
                         f"Sampling rate: {sfreq:.0f} Hz")
            self.file_info_label.config(text=info_text)
            
            self.time_end_var.set(min(10.0, duration))
            self.time_start_spin.config(to=duration)
            self.time_end_spin.config(to=duration)
            
            self.update_status("File loaded successfully. Ready to process.")
            self.progress['value'] = 0
            self.process_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self.progress['value'] = 0
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            self.update_status("Failed to load file.")
            
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update()
        
    def update_progress(self, value, message=""):
        """Update progress bar"""
        self.progress['value'] = value
        if message:
            self.update_status(message)
        self.root.update()
        
    def stop_processing(self):
        """Stop processing"""
        self.stop_requested = True
        self.stop_button.config(state=tk.DISABLED)
        
    def run_reconstruction(self):
        """Run source reconstruction in a separate thread"""
        self.stop_requested = False
        self.stop_button.config(state=tk.NORMAL)
        self.process_button.config(state=tk.DISABLED)
        
        self.results_text.delete(1.0, tk.END)
        self.notebook.select(self.results_frame)
        
        self.processing_thread = threading.Thread(target=self._process_data)
        self.processing_thread.start()
        
    def _process_data(self):
        """Main processing function (runs in separate thread)"""
        try:
            # Step 1: Check brain template
            self.update_progress(10, "Step 1/8: Checking brain template...")
            subjects_dir = self._check_fsaverage()
            if self.stop_requested: return
            
            # Step 2: Preprocessing
            self.update_progress(20, "Step 2/8: Preprocessing EEG data...")
            raw_processed = self._preprocess_raw()
            if self.stop_requested: return
            
            # Step 3: Create forward solution
            self.update_progress(40, "Step 3/8: Creating forward solution...")
            fwd = self._create_forward_solution(raw_processed, subjects_dir)
            if self.stop_requested: return
            
            # Step 4: Compute inverse operator
            self.update_progress(50, "Step 4/8: Computing inverse operator...")
            inverse_operator = self._compute_inverse_operator(raw_processed, fwd)
            if self.stop_requested: return
            
            mode = self.analysis_mode.get()

            if mode == "standard":
                # Step 5: Filter for selected frequency band
                self.update_progress(60, "Step 5/8: Filtering for standard analysis...")
                raw_filtered, freq_band_name = self._filter_for_standard_analysis(raw_processed)
                if self.stop_requested: return
                
                # Step 6: Reconstruct sources
                self.update_progress(70, "Step 6/8: Reconstructing sources...")
                stc = self.reconstructor.reconstruct(raw_filtered, inverse_operator, self.method_var.get())
                self.log_result(f"Applied {self.method_var.get()} inverse solution.")

                # Step 7: Process visualization
                self.update_progress(80, "Step 7/8: Processing standard visualization...")
                stc_viz, params = self._process_standard_visualization(stc, freq_band_name)

            elif mode == "coordination":
                # Step 5: Reconstruct BROADBAND sources first
                self.update_progress(60, "Step 5/8: Reconstructing broadband sources for coordination model...")
                stc_broadband = self.reconstructor.reconstruct(raw_processed, inverse_operator, self.method_var.get())
                if self.stop_requested: return

                # Step 6: Run coordination analysis in source space
                self.update_progress(70, "Step 6/8: Calculating coordination dynamics...")
                coord_metrics = self._analyze_coordination_in_source_space(stc_broadband)
                if self.stop_requested: return

                # Step 7: Select the metric to visualize
                self.update_progress(80, "Step 7/8: Processing coordination visualization...")
                metric_to_show = self.coord_viz_var.get()
                stc_viz, params = self._process_coordination_visualization(coord_metrics, metric_to_show)

            # Step 8: Create visualization
            self.update_progress(95, "Step 8/8: Creating brain visualization...")
            self.root.after(0, lambda sv=stc_viz, p=params, sd=subjects_dir: 
                            self._create_visualization(sv, p, sd))
            
            self.update_progress(100, "Processing complete!")
            self.log_result("\n✓ Source reconstruction completed successfully!")
            
        except Exception as e:
            self.update_progress(0, f"Error: {str(e)}")
            self.log_result(f"\n✗ Error during processing: {str(e)}")
            # --- FIX: Bind the exception 'e' to the lambda's scope ---
            self.root.after(0, lambda err=e: messagebox.showerror("Processing Error", str(err)))
            
        finally:
            self.process_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def _preprocess_raw(self):
        """Handles the full preprocessing pipeline."""
        raw_processed = self.raw.copy()
        
        # Clean channel names
        cleaned_names = {name: name.strip().replace('.', '').upper() for name in raw_processed.ch_names}
        raw_processed.rename_channels(cleaned_names, verbose=False)
        
        # Set montage
        montages_to_try = ['standard_1005', 'standard_1020', 'biosemi64', 'biosemi32']
        montage_set = False
        for montage_name in montages_to_try:
            try:
                montage = mne.channels.make_standard_montage(montage_name)
                raw_processed.set_montage(montage, match_case=False, on_missing='ignore', verbose=False)
                if any(ch['loc'][0] for ch in raw_processed.info['chs'] if not np.isnan(ch['loc'][0])):
                    montage_set = True
                    self.log_result(f"Successfully applied {montage_name} montage.")
                    break
            except Exception:
                continue
        if not montage_set:
            raise ValueError("Could not apply any standard montage. Please check channel names.")

        # Bad channel detection and interpolation
        if self.remove_bad_channels_var.get():
            bad_channels = self.preprocessing.detect_bad_channels(raw_processed)
            if bad_channels:
                self.log_result(f"Detected bad channels: {', '.join(bad_channels)}")
                raw_processed.info['bads'] = bad_channels
                raw_processed.interpolate_bads(reset_bads=True, verbose=False)
        
        # Artifact removal
        raw_processed = self.preprocessing.remove_artifacts(raw_processed, self.artifact_removal_var.get())
        
        # Set reference
        raw_processed.set_eeg_reference('average', projection=True, verbose=False)
        
        # Crop to selected time window
        tmin, tmax = self.time_start_var.get(), self.time_end_var.get()
        raw_processed.crop(tmin=tmin, tmax=tmax)
        self.log_result(f"Analyzing time window: {tmin}-{tmax} seconds")
        
        return raw_processed

    def _filter_for_standard_analysis(self, raw_processed):
        """Filters data for standard analysis mode."""
        freq_band_name = self.get_frequency_band()
        if freq_band_name != 'broadband':
            low_freq, high_freq = self.freq_bands[freq_band_name]
            nyquist = raw_processed.info['sfreq'] / 2.0
            if high_freq >= nyquist:
                high_freq = nyquist - 1
                self.log_result(f"Adjusted high frequency to {high_freq:.1f} Hz due to Nyquist limit.")
            raw_processed.filter(low_freq, high_freq, fir_design='firwin', verbose=False)
            self.log_result(f"Filtered to {freq_band_name} band: {low_freq}-{high_freq:.1f} Hz")
        return raw_processed, freq_band_name
    
    def _analyze_coordination_in_source_space(self, stc_broadband):
        """
        Calculates Universal Brain Coordination metrics, including the new 'Coordinated Power' (Y*PLV).
        """
        self.log_result("Analyzing coordination dynamics in source space...")
        
        conductor_band_name = self.conductor_var.get()
        conductor_freqs = self.freq_bands[conductor_band_name]
        
        orchestra_freqs = [self.freq_bands[band] for band, var in self.orchestra_vars.items() if var.get() and band != conductor_band_name]
        if not orchestra_freqs: raise ValueError("Orchestra must contain bands different from the Conductor.")
        min_orch_freq, max_orch_freq = min(f[0] for f in orchestra_freqs), max(f[1] for f in orchestra_freqs)
        
        stc_conductor = stc_broadband.copy().filter(l_freq=conductor_freqs[0], h_freq=conductor_freqs[1], verbose=False)
        stc_moire = stc_broadband.copy().filter(l_freq=min_orch_freq, h_freq=max_orch_freq, verbose=False)
        
        # Calculate intermediate data
        y_data = stc_moire.data ** 2
        phase_conductor = np.angle(hilbert(stc_conductor.data, axis=1))
        phase_moire = np.angle(hilbert(stc_moire.data, axis=1))
        
        # This is the Phase-Locking Value (PLV), which represents COORDINATION
        plv_instantaneous = np.abs(np.exp(1j * (phase_conductor - phase_moire)))
        
        # This is the Phase-Slip Rate (Z-Axis)
        z_data = 1 - plv_instantaneous
        
        # --- THE CORRECTED METRIC ---
        # Multiply orchestra power by the coordination (PLV), not the slip rate
        coordinated_power_data = y_data * plv_instantaneous

        # --- FIX: Create new SourceEstimate objects using attributes from the original ---
        # MNE SourceEstimate objects don't have a .get_params() method.
        # We pass the metadata (vertices, timing info) from the original stc.
        stc_x = mne.SourceEstimate(stc_conductor.data ** 2, vertices=stc_broadband.vertices, tmin=stc_broadband.tmin, tstep=stc_broadband.tstep, subject=stc_broadband.subject)
        stc_y = mne.SourceEstimate(y_data, vertices=stc_broadband.vertices, tmin=stc_broadband.tmin, tstep=stc_broadband.tstep, subject=stc_broadband.subject)
        stc_z = mne.SourceEstimate(z_data, vertices=stc_broadband.vertices, tmin=stc_broadband.tmin, tstep=stc_broadband.tstep, subject=stc_broadband.subject)
        stc_cp = mne.SourceEstimate(coordinated_power_data, vertices=stc_broadband.vertices, tmin=stc_broadband.tmin, tstep=stc_broadband.tstep, subject=stc_broadband.subject)
        
        self.log_result("✓ Time-resolved coordination analysis complete.")
        return {
            'Conductor Power': stc_x, 
            'Moiré Harmony': stc_y, 
            'Phase-Slip Rate': stc_z, 
            'Coordinated Power (Y*PLV)': stc_cp
        }

    def _process_standard_visualization(self, stc, freq_band_name):
        """Process visualization for standard analysis"""
        viz_type = self.viz_var.get()
        title = f"{viz_type.title()} - {freq_band_name.title()} Band"
        
        # --- FIX: Replaced all instances of **stc.get_params() ---
        stc_meta = {'vertices': stc.vertices, 'tmin': stc.tmin, 'tstep': stc.tstep, 'subject': stc.subject}

        if viz_type == "phase":
            phase_data = np.angle(hilbert(stc.data, axis=1))
            stc_viz = mne.SourceEstimate(phase_data, **stc_meta)
            params = {'colormap': 'twilight_shifted', 'clim': dict(kind='value', lims=[-np.pi, 0, np.pi]), 'title': title}
        elif viz_type == "power":
            power_data = stc.data ** 2
            stc_viz = mne.SourceEstimate(power_data, **stc_meta)
            params = {'colormap': 'hot', 'clim': dict(kind='percent', lims=[90, 95, 99]), 'title': title}
        elif viz_type == "stats":
            z_scores = (stc.data - np.mean(stc.data)) / np.std(stc.data)
            stc_viz = mne.SourceEstimate(z_scores, **stc_meta)
            params = {'colormap': 'RdBu_r', 'clim': dict(kind='value', lims=[-2.5, 0, 2.5]), 'title': title}
        else: # raw
            stc_viz = stc
            params = {'colormap': 'RdBu_r', 'clim': dict(kind='percent', lims=[5, 50, 95]), 'title': title}
        
        return stc_viz, params
    
    def _process_coordination_visualization(self, coord_metrics, metric_to_show):
        """Process visualization for coordination analysis with robust color scaling."""
        stc_viz = coord_metrics[metric_to_show]
        title = f"Coordination Model: {metric_to_show}"
        
        if metric_to_show == 'Conductor Power' or metric_to_show == 'Moiré Harmony':
            # Use percentile scaling for power maps
            params = {
                'colormap': 'hot', 
                'clim': dict(kind='percent', lims=[90, 95, 99]), 
                'title': title
            }
        elif metric_to_show == 'Phase-Slip Rate':
            # Use percentile scaling to guarantee colors are always visible.
            params = {
                'colormap': 'viridis', 
                'clim': dict(kind='percent', lims=[1, 50, 99]), 
                'title': title
            }
        # --- CHANGE 3: CORRECTED LOGIC --- This handles visualization for the new Y*PLV metric
        elif 'Coordinated Power' in metric_to_show:
             params = {
                 'colormap': 'plasma', # A good colormap for power+coordination
                 'clim': dict(kind='percent', lims=[95, 97, 99.9]), # Focus on the top hotspots
                 'title': title
             }
        
        return stc_viz, params
        
    def _create_visualization(self, stc, params, subjects_dir):
        """Create brain visualization (must be called on main thread)"""
        try:
            time_label = f"{params['title']}"
            if stc.data.ndim > 1 and stc.data.shape[1] > 1:
                 time_label += " (t=%0.2f s)"

            brain = stc.plot(
                subjects_dir=subjects_dir, subject='fsaverage', surface='pial',
                hemi='both', colormap=params['colormap'], clim=params['clim'],
                time_label=time_label, size=(1000, 750),
                smoothing_steps=5, background='white', verbose=False
            )
            self.brain_figures.append(brain)
            self.log_result(f"Created 3D visualization with {mne.viz.get_3d_backend()}")
        except Exception as e:
            self.log_result(f"3D visualization error: {str(e)}")

    def _create_forward_solution(self, raw, subjects_dir):
            """Create forward solution using the correct 3-layer BEM for EEG."""
            # Create a 3-layer BEM model suitable for EEG
            self.log_result("Creating 3-layer BEM model (brain, skull, scalp)...")
            model = mne.make_bem_model(
                subject='fsaverage', 
                ico=4,
                conductivity=(0.3, 0.006, 0.3),  # Correct 3-layer conductivity
                subjects_dir=subjects_dir, 
                verbose=False
            )
            bem_sol = mne.make_bem_solution(model, verbose=False)
            
            # Setup source space
            spacing = self.spacing_var.get()
            src = mne.setup_source_space(
                'fsaverage', 
                spacing=spacing,
                add_dist=False,
                subjects_dir=subjects_dir, 
                verbose=False
            )
            self.log_result(f"Created source space with {spacing} spacing")
            
            # Make forward solution
            fwd = mne.make_forward_solution(
                raw.info, 
                trans='fsaverage',
                src=src, 
                bem=bem_sol,
                meg=False, 
                eeg=True,
                mindist=5.0, 
                verbose=False
            )
            
            return fwd
        
    def _compute_inverse_operator(self, raw, fwd):
        """Compute inverse operator"""
        noise_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None, verbose=False)
        inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=False)
        self.log_result("Created inverse operator")
        return inverse_operator

    def _check_fsaverage(self):
            """
            Checks for the 'fsaverage' brain template and downloads it if missing.
            This version is more robust and avoids relying on a pre-configured environment.
            """
            # Define a standard, reliable path for MNE data in the user's home directory.
            subjects_dir = os.path.join(os.path.expanduser('~'), 'mne_data')
            
            # Define the expected full path to the 'fsaverage' template directory.
            fsaverage_path = os.path.join(subjects_dir, 'fsaverage')
            
            # Check if the fsaverage directory exists. If not, download it to our defined path.
            if not os.path.isdir(fsaverage_path):
                self.log_result("Fsaverage brain template not found.")
                self.log_result(f"Downloading to: {subjects_dir} (this may take a few minutes)...")
                
                # This function will download and place the data in the correct subdirectories.
                # It is the most reliable way to ensure the template is available.
                mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)
                self.log_result("✓ Fsaverage download complete.")
            else:
                self.log_result("✓ Fsaverage brain template found.")

            # Always return this reliable path for other functions to use.
            return subjects_dir

    def get_frequency_band(self):
        """Get selected frequency band"""
        return self.freq_var.get()
    
    @property
    def freq_bands(self):
        """Frequency band definitions"""
        return {
            "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12),
            "beta": (12, 30), "gamma": (30, 50), "broadband": (0.5, 50)
        }
        
    def close_brain_views(self):
        """Close all open brain visualization windows"""
        for brain in self.brain_figures:
            try: brain.close()
            except: pass
        self.brain_figures.clear()
        self.log_result("Closed all 3D brain views")
        
    def log_result(self, message):
        """Log message to results tab"""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
        self.root.update()

if __name__ == "__main__":
    # Set up 3D backend
    try:
        mne.viz.set_3d_backend("pyvistaqt")
    except Exception:
        print("PyVistaQt backend not available, trying others.")
    
    # Create and run application
    root = tk.Tk()
    app = EEGSourceReconstructionApp(root)
    root.mainloop()