import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import mne
from scipy.signal import butter, lfilter
from tqdm import tqdm
import gradio as gr
import plotly.express as px
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- (Configuration and Model classes are unchanged) ---
class Config:
    def __init__(self):
        self.fs = 100.0
        self.epoch_length = 1.0
        self.frequency_bands = {
            'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
            'beta': (13, 30), 'gamma': (30, 49)
        }
        self.latent_dim = 32
        self.eeg_batch_size = 64
        self.eeg_epochs = 25
        self.learning_rate = 1e-3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EEGAutoencoder(nn.Module):
    def __init__(self, channels=64, frequency_bands=5, latent_dim=32):
        super(EEGAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * frequency_bands, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, channels * frequency_bands),
            nn.Sigmoid()
        )
        self.channels = channels
        self.frequency_bands = frequency_bands

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        reconstruction = reconstruction.view(-1, self.channels, self.frequency_bands)
        return reconstruction, latent

# --- (EEG_REGIONS definition is unchanged) ---
EEG_REGIONS = {
    "Occipital": ['O1', 'O2', 'Oz', 'POz', 'PO3', 'PO4', 'PO7', 'PO8'],
    "Temporal": ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8', 'T9', 'T10'],
    "Parietal": ['P1', 'P2', 'P3', 'P4', 'Pz', 'CP1', 'CP2', 'CPz'],
    "Frontal": ['Fp1', 'Fp2', 'Fz', 'F1', 'F2', 'F3', 'F4', 'F7', 'F8'],
    "Central": ['C1', 'C2', 'C3', 'C4', 'Cz', 'FC1', 'FC2', 'FCz'],
    "All": []
}

### --- NEW: EEG-SPECIFIC BAD CHANNEL DETECTION --- ###
def detect_bad_channels_eeg(raw, threshold=3.0):
    """
    Simple bad channel detection for EEG based on variance.
    Adapted from the logic in mnebrain_signalvs_composite3.py.
    """
    data = raw.get_data(picks='eeg')
    channel_variances = np.var(data, axis=1)
    
    # Use median absolute deviation for robust z-scoring
    median_var = np.median(channel_variances)
    mad = np.median(np.abs(channel_variances - median_var))
    
    if mad == 0:
        return [] # Avoid division by zero if all variances are the same
        
    # Robust z-score calculation (0.6745 is the 75th percentile of the standard normal distribution)
    z_scores = 0.6745 * (channel_variances - median_var) / mad
    bad_indices = np.where(np.abs(z_scores) > threshold)[0]
    
    bad_channel_names = [raw.ch_names[i] for i in bad_indices]
    return bad_channel_names

def robust_preprocess_eeg(raw):
    """Applies a robust preprocessing pipeline to the raw EEG data."""
    # 1. Standardize channel names
    raw.rename_channels(lambda name: name.strip().replace('.', '').upper(), allow_duplicates=True)

    # 2. Set a standard montage
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore', match_case=False, verbose=False)

    ### --- MODIFIED: USE THE CORRECT FUNCTION --- ###
    # 3. Detect and interpolate bad channels using the EEG-specific method
    bad_channels = detect_bad_channels_eeg(raw)
    if bad_channels:
        print(f"Found bad channels: {bad_channels}")
        raw.info['bads'] = bad_channels
        raw.interpolate_bads(reset_bads=True, verbose=False)

    # 4. Set average reference
    raw.set_eeg_reference('average', projection=True, verbose=False)
    raw.apply_proj()

    return raw

# --- (Other functions: epoch_and_extract_power, EEGDataset, analyze_cluster_band_power are unchanged) ---
def epoch_and_extract_power(raw, config):
    raw.resample(config.fs)
    samples_per_epoch = int(config.epoch_length * config.fs)
    n_epochs = len(raw) // samples_per_epoch
    
    all_band_powers = []
    for i in tqdm(range(n_epochs), desc="Epoching and Filtering"):
        epoch_data = raw.get_data(start=i*samples_per_epoch, stop=(i+1)*samples_per_epoch)
        epoch_band_powers = []
        for band, (low, high) in config.frequency_bands.items():
            filtered = mne.filter.filter_data(epoch_data, sfreq=config.fs, l_freq=low, h_freq=high, verbose=False)
            power = np.mean(filtered**2, axis=1)
            epoch_band_powers.append(power)
        all_band_powers.append(np.stack(epoch_band_powers, axis=1))
    data = np.array(all_band_powers)
    
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    std[std == 0] = 1
    normalized_data = (data - mean) / (std * 4) + 0.5 
    np.clip(normalized_data, 0, 1, out=normalized_data)
    
    return normalized_data

def analyze_cluster_band_power(original_data, cluster_labels, band_names):
    n_clusters = len(np.unique(cluster_labels))
    analysis_results = {}
    avg_power_per_epoch = np.mean(original_data, axis=1)
    
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) == 0: continue
        power_for_cluster = avg_power_per_epoch[cluster_indices]
        mean_band_power = np.mean(power_for_cluster, axis=0)
        
        cluster_name = f"Cluster {i}"
        analysis_results[cluster_name] = {
            "epoch_count": len(cluster_indices),
            "dominant_band": band_names[np.argmax(mean_band_power)],
            "band_powers": {band: f"{power:.4f}" for band, power in zip(band_names, mean_band_power)}
        }
    df_data = []
    for cluster, data in analysis_results.items():
        row = {'Cluster': cluster, 'Epochs': data['epoch_count'], 'Dominant Band': data['dominant_band']}
        row.update({f"{k.capitalize()} Power": v for k, v in data['band_powers'].items()})
        df_data.append(row)
    return pd.DataFrame(df_data)
    
class EEGDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- (Main Gradio run function and interface are unchanged) ---
def run_full_analysis(edf_file, region_to_analyze, n_clusters, reduction_method, progress=gr.Progress()):
    if edf_file is None:
        raise gr.Error("Please upload an EDF file.")
    config = Config()
    progress(0.1, desc="Loading and applying robust preprocessing...")
    raw = mne.io.read_raw_edf(edf_file.name, preload=True, verbose=False)
    raw = robust_preprocess_eeg(raw)

    if region_to_analyze != "All":
        region_channels = EEG_REGIONS[region_to_analyze]
        available_channels = [ch for ch in region_channels if ch in raw.ch_names]
        if not available_channels:
             raise gr.Error(f"File contains no standard channels for the {region_to_analyze} region.")
        raw.pick_channels(available_channels)

    original_data = epoch_and_extract_power(raw, config)

    progress(0.3, desc="Training Autoencoder...")
    dataset = EEGDataset(original_data)
    dataloader = DataLoader(dataset, batch_size=config.eeg_batch_size, shuffle=True)
    
    n_channels, n_bands = original_data.shape[1], original_data.shape[2]
    model = EEGAutoencoder(channels=n_channels, frequency_bands=n_bands, latent_dim=config.latent_dim).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in progress.tqdm(range(config.eeg_epochs), desc="Training"):
        for data_batch in dataloader:
            data_batch = data_batch.to(config.device)
            optimizer.zero_grad()
            recon, latent = model(data_batch)
            loss = criterion(recon, data_batch)
            loss.backward()
            optimizer.step()
            
    progress(0.7, desc="Extracting, clustering, and analyzing...")
    model.eval()
    all_latents = []
    full_dataloader = DataLoader(dataset, batch_size=config.eeg_batch_size, shuffle=False)
    with torch.no_grad():
        for data_batch in full_dataloader:
            _, latent = model(data_batch.to(config.device))
            all_latents.append(latent.cpu().numpy())
    hidden_vectors = np.vstack(all_latents)
    
    if reduction_method == 'UMAP': reducer = umap.UMAP(n_components=3, random_state=42)
    else: reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(hidden_vectors)-1))
    reduced_vectors = reducer.fit_transform(hidden_vectors)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(hidden_vectors)
    
    correlation_df = analyze_cluster_band_power(original_data, cluster_labels, list(config.frequency_bands.keys()))
    
    plot_df = pd.DataFrame({'x': reduced_vectors[:, 0], 'y': reduced_vectors[:, 1], 'z': reduced_vectors[:, 2], 'cluster': [f"Cluster {l}" for l in cluster_labels]})
    fig = px.scatter_3d(plot_df, x='x', y='y', z='z', color='cluster', title=f"Latent Space of {region_to_analyze} Brain States ({reduction_method})", template='plotly_dark')
    fig.update_traces(marker=dict(size=3, opacity=0.8))
    
    return fig, correlation_df

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# Latent Space Conductor Analyzer (v3 - Robust EEG)")
    with gr.Row():
        with gr.Column(scale=1):
            edf_input = gr.File(label="Upload EEG File (.edf)")
            region_input = gr.Dropdown(choices=list(EEG_REGIONS.keys()), value="Occipital", label="Select Brain Region to Analyze")
            cluster_input = gr.Slider(minimum=2, maximum=15, value=5, step=1, label="Number of Brain States (Clusters)")
            reduction_input = gr.Radio(choices=["UMAP", "t-SNE"], value="UMAP", label="3D Visualization Method")
            run_button = gr.Button("Discover Brain States", variant="primary")
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Discovered Brain States (3D Latent Space)")
            df_output = gr.DataFrame(label="Analysis of Discovered States (Hypothesis Derivation)")
    run_button.click(
        fn=run_full_analysis,
        inputs=[edf_input, region_input, cluster_input, reduction_input],
        outputs=[plot_output, df_output]
    )

if __name__ == "__main__":
    app.launch(debug=True)