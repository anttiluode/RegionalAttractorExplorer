import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import mne
import gradio as gr
import plotly.graph_objects as go
from tqdm import tqdm

# --- ARCHITECTURE AND TRAINING CLASSES (Unchanged) ---
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, latent_dim))
    def forward(self, x): return self.net(x)

class Predictor(nn.Module):
    def __init__(self, latent_dim=32, depth=3):
        super().__init__()
        layers = [nn.Linear(latent_dim, 128), nn.ReLU()]
        for _ in range(depth - 1): layers.extend([nn.Linear(128, 128), nn.ReLU()])
        layers.append(nn.Linear(128, latent_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, z): return self.net(z)

class WorldModel(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.predictor = Predictor(latent_dim)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.pair_dist = nn.PairwiseDistance(p=2)
    def forward(self, pred, pos, neg):
        pos_dist = self.pair_dist(pred, pos)
        neg_dist = self.pair_dist(pred, neg)
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return torch.mean(loss)

class PairedEEGDataset(Dataset):
    def __init__(self, epoch_data): self.data = epoch_data
    def __len__(self): return len(self.data) - 1
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.data[idx+1], dtype=torch.float32)

EEG_REGIONS = {"All": [], "Occipital": ['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4', 'PO7', 'PO8'], "Temporal": ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8', 'T9', 'T10'], "Parietal": ['P1', 'P2', 'P3', 'P4', 'PZ', 'CP1', 'CP2', 'CPZ'], "Frontal": ['FP1', 'FP2', 'FZ', 'F1', 'F2', 'F3', 'F4', 'F7', 'F8'], "Central": ['C1', 'C2', 'C3', 'C4', 'CZ', 'FC1', 'FC2', 'FCZ']}

# ==================================
# 2. DATA HANDLING - THE CRITICAL FIX
# ==================================

### --- START OF UPGRADE --- ###
def create_eeg_features(edf_file, region="All", epoch_sec=0.5, fs=100.0):
    """Correctly extracts features by filtering the whole signal BEFORE epoching."""
    
    frequency_bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    raw.rename_channels(lambda name: name.strip().replace('.', '').upper())
    
    if region != "All":
        region_channels = EEG_REGIONS[region]
        available_channels = [ch for ch in region_channels if ch in raw.ch_names]
        if not available_channels: raise gr.Error(f"File has no standard channels for {region} region.")
        raw.pick_channels(available_channels)

    raw.resample(fs, verbose=False)

    # STEP 1: Filter the ENTIRE continuous signal for each band
    band_filtered_data = {}
    for band, (low, high) in frequency_bands.items():
        band_filtered_data[band] = raw.copy().filter(l_freq=low, h_freq=high, fir_design='firwin', verbose=False).get_data()

    # STEP 2: NOW, epoch the pre-filtered signals and calculate power
    samples_per_epoch = int(epoch_sec * fs)
    all_epochs_features = []
    
    for i in range(0, raw.n_times - samples_per_epoch, samples_per_epoch):
        epoch_band_powers = []
        for band in frequency_bands.keys():
            # Get the window from the already-filtered data
            epoch_data = band_filtered_data[band][:, i:i+samples_per_epoch]
            power = np.log1p(np.mean(epoch_data**2, axis=1))
            epoch_band_powers.append(power)
        
        epoch_features = np.stack(epoch_band_powers, axis=1)
        all_epochs_features.append(epoch_features)
        
    all_epochs_features = np.array(all_epochs_features)
    n_epochs, n_channels, n_bands = all_epochs_features.shape
    flattened_features = all_epochs_features.reshape(n_epochs, n_channels * n_bands)
    
    mean = np.mean(flattened_features, axis=0, keepdims=True)
    std = np.std(flattened_features, axis=0, keepdims=True)
    std[std == 0] = 1
    normalized_features = (flattened_features - mean) / std
    
    return normalized_features, normalized_features.shape[1]
### --- END OF UPGRADE --- ###

# ==================================
# 3. VISUALIZATION - THE PLOTTING FIX
# ==================================

def visualize_energy_landscape(model, latent_dim, device, region):
    model.eval()
    with torch.no_grad():
        n_points = 30
        # Use a wider range to see potentially more complex landscapes
        bounds = np.abs(model.encoder.net[-1].weight.data.cpu().numpy()).max() * 2 
        x_range, y_range = np.linspace(-bounds, bounds, n_points), np.linspace(-bounds, bounds, n_points)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
        if latent_dim > 2: grid_points = np.hstack([grid_points, np.zeros((grid_points.shape[0], latent_dim - 2))])
        
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
        predictions = model.predictor(grid_tensor)
        energy = torch.linalg.norm(predictions - grid_tensor, axis=1).cpu().numpy()
        energy_grid = energy.reshape(n_points, n_points)

    fig = go.Figure(data=[go.Surface(z=energy_grid, x=x_range, y=y_range, colorscale='viridis_r', cmin=np.min(energy_grid), cmax=np.percentile(energy_grid, 95))])
    fig.update_layout(title=f"Proof 1: Energy Landscape of {region} Lobe", scene=dict(xaxis_title='Latent Dim 1', yaxis_title='Latent Dim 2', zaxis_title='Prediction Energy'), template='plotly_dark')
    return fig

### --- START OF PLOTTING FIX --- ###
def visualize_predicted_trajectory(model, epoch_data, device, region):
    model.eval()
    with torch.no_grad():
        full_sequence = torch.tensor(epoch_data, dtype=torch.float32).to(device)
        true_trajectory = model.encoder(full_sequence).cpu().numpy()
        predicted_trajectory = [true_trajectory[0]]
        current_z = true_trajectory[0]
        for _ in range(len(true_trajectory) - 1):
            current_z_tensor = torch.tensor(current_z, dtype=torch.float32).unsqueeze(0).to(device)
            next_z = model.predictor(current_z_tensor).squeeze(0).cpu().numpy()
            predicted_trajectory.append(next_z)
            current_z = next_z
        predicted_trajectory = np.array(predicted_trajectory)

    fig = go.Figure()
    latent_dim = true_trajectory.shape[1]

    # Gracefully handle 2D or 3D plotting
    if latent_dim >= 3:
        fig.add_trace(go.Scatter3d(x=true_trajectory[:,0], y=true_trajectory[:,1], z=true_trajectory[:,2], mode='lines+markers', name='Actual Trajectory', line=dict(color='cyan', width=4)))
        fig.add_trace(go.Scatter3d(x=predicted_trajectory[:,0], y=predicted_trajectory[:,1], z=predicted_trajectory[:,2], mode='lines+markers', name='Predicted Trajectory', line=dict(color='magenta', width=4, dash='dot')))
    else: # latent_dim == 2
        fig.add_trace(go.Scatter(x=true_trajectory[:,0], y=true_trajectory[:,1], mode='lines+markers', name='Actual Trajectory', line=dict(color='cyan', width=4)))
        fig.add_trace(go.Scatter(x=predicted_trajectory[:,0], y=predicted_trajectory[:,1], mode='lines+markers', name='Predicted Trajectory', line=dict(color='magenta', width=4, dash='dot')))
    
    fig.update_layout(title=f"Proof 2: Predicted Future of {region} Lobe States", template='plotly_dark')
    return fig
### --- END OF PLOTTING FIX --- ###


# ==================================
# 4. GRADIO APP (Unchanged)
# ==================================
def run_experiment(edf_file, region, latent_dim, epochs, progress=gr.Progress()):
    if edf_file is None: raise gr.Error("Upload an EEG file.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    progress(0.1, desc=f"Extracting multi-band features for {region}...")
    epoch_data, input_dim = create_eeg_features(edf_file.name, region)
    dataset = PairedEEGDataset(epoch_data)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = WorldModel(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = ContrastiveLoss()
    
    progress(0.3, desc=f"Training World Model for {region}...")
    model.train()
    for epoch in progress.tqdm(range(epochs), desc="Training Epochs"):
        for x_t, x_t_plus_1 in dataloader:
            x_t, x_t_plus_1, device = x_t.to(device), x_t_plus_1.to(device), model.encoder.net[0].weight.device
            optimizer.zero_grad()
            z_t, z_pos = model.encoder(x_t), model.encoder(x_t_plus_1)
            z_neg = z_pos[torch.randperm(z_pos.size(0))]
            z_pred = model.predictor(z_t)
            loss = criterion(z_pred, z_pos, z_neg)
            loss.backward()
            optimizer.step()

    progress(0.8, desc="Generating proof visualizations...")
    fig1 = visualize_energy_landscape(model, latent_dim, device, region)
    fig2 = visualize_predicted_trajectory(model, epoch_data[:200], device, region)
    return fig1, fig2

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# The Hurricane Model (v3 - Final Fix)")
    gr.Markdown("This version uses a **correct signal processing pipeline** (Filter First, Then Epoch) to provide the AI with clean, meaningful features.")
    with gr.Row():
        with gr.Column(scale=1):
            edf_input = gr.File(label="Upload EEG File (.edf)")
            region_selector = gr.Dropdown(choices=list(EEG_REGIONS.keys()), value="All", label="Region Selector (Zoom Lens)")
            latent_dim_slider = gr.Slider(2, 64, value=8, step=1, label="Latent Dimensions")
            epochs_slider = gr.Slider(10, 200, value=50, step=1, label="Training Epochs")
            run_button = gr.Button("Build and Prove World Model", variant="primary")
        with gr.Column(scale=2):
            gr.Markdown("## The Proof:")
            plot_landscape = gr.Plot(label="Proof 1: Learned Energy Landscape")
            plot_trajectory = gr.Plot(label="Proof 2: Predicted Future vs. Reality")
    run_button.click(fn=run_experiment, inputs=[edf_input, region_selector, latent_dim_slider, epochs_slider], outputs=[plot_landscape, plot_trajectory])

if __name__ == "__main__":
    app.launch(debug=True)