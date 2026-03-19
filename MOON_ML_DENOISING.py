
import os, time, warnings
import numpy as np
import spectral
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
HDR_PATH    = r"D:\Moon_Data\Scene_2\Physics_Corrected\M3G20081201T064047_V01_RFL_CORRECTED.hdr"
OUT_DIR     = r"D:\Moon_Data\Scene_2\ML_Denoised"
OUT_HDR     = os.path.join(OUT_DIR, "M3G20081201T064047_V01_RFL_FINAL.hdr")
PROOF_FIG   = os.path.join(OUT_DIR, "ml_denoising_proof.png")

# ML Weight Path (for Scene Generalisation)
AE_WEIGHTS  = r"d:\Moon\m3_autoencoder.pth"

# Thresholds
THR_NER   = 0.5     # %
THR_SPIKE = 1.0     # %
THR_SNR   = 100.0   # ratio

# ML Hyperparameters
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_STR  = 50    # CNN Destriper epochs
EPOCHS_AE   = 40    # Autoencoder epochs
BATCH_SIZE  = 512
AE_DIM      = 8     # bottleneck dimension
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

def banner(msg): print("\n" + "="*70 + f"\n  {msg}\n" + "="*70)

# ── 1. NOISE MEASUREMENT ──────────────────────────────────────────────────────
def measure_noise(cube):
    rows, cols, bands = cube.shape
    
    # 1. Stripe (NER)
    ner_list = []
    for b in range(bands):
        bd = cube[:, :, b]
        valid = np.isfinite(bd)
        if valid.sum() < 50: continue
        col_means = np.nanmean(bd, axis=0)
        scene_mean = np.nanmean(bd[valid])
        if scene_mean > 0: ner_list.append(np.std(col_means) / scene_mean * 100)
    ner_val = float(np.median(ner_list)) if ner_list else 0.0

    # 2. Gaussian (SNR)
    snr_list = []
    for b in range(bands):
        bd = cube[:, :, b]
        valid = np.isfinite(bd)
        if valid.sum() < 50: continue
        smooth = ndimage.uniform_filter(np.nan_to_num(bd, nan=0.0), size=3)
        noise = np.nanstd((bd - smooth)[valid])
        sig = np.nanmean(bd[valid])
        if noise > 0: snr_list.append(sig / noise)
    snr_val = float(np.median(snr_list)) if snr_list else 0.0

    # 3. Spikes (MAD Z-score)
    step = max(1, rows // 50)
    total_px = 0; spike_px = 0
    for ri in range(0, rows, step):
        for ci in range(0, cols, 4):
            spec = cube[ri, ci, :]
            if not np.any(np.isfinite(spec)): continue
            total_px += 1
            med = np.nanmedian(spec)
            mad = np.nanmedian(np.abs(spec - med))
            if mad > 1e-10 and np.any(np.abs(0.6745*(spec-med)/mad) > 3.5):
                spike_px += 1
    spike_val = (spike_px / total_px * 100) if total_px > 0 else 0.0

    return ner_val, snr_val, spike_val

# ── 2. ML MODELS ──────────────────────────────────────────────────────────────
class CNNDestriper(nn.Module):
    """1D Spatial CNN: Learns column variation (stripe pattern) per band."""
    def __init__(self, cols):
        super().__init__()
        # Input: (batch=rows, channels=1, length=cols)
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 1, kernel_size=1)
        )
    def forward(self, x):
        # Neural net estimates the stripe noise, we subtract it from x
        stripe_noise = self.net(x)
        return x - stripe_noise

class SpectralAutoencoder(nn.Module):
    """Undercomplete AE: Compresses spectral manifold, discarding Gaussian noise."""
    def __init__(self, bands, hidden=AE_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(bands, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, hidden)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, bands)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
banner("STEP 0  ·  Loading Corrected M3 L2 Cube")
if not os.path.exists(HDR_PATH):
    raise FileNotFoundError(f"Input cube not found: {HDR_PATH}\nRun Physics Corrections first.")

img = spectral.open_image(HDR_PATH)
in_cube = np.array(img.load(), dtype=np.float32)
rows, cols, bands = in_cube.shape
in_cube[in_cube < -990] = np.nan
print(f"  Shape: {rows} × {cols} × {bands}")
print(f"  Using device: {DEVICE}")

wl = np.array([float(w) for w in img.metadata.get('wavelength', [])])
if wl.max() < 10: wl *= 1000.0

banner("STEP 1  ·  Initial Noise Check")
ner_0, snr_0, spike_0 = measure_noise(in_cube)
print(f"  Stripe NER : {ner_0:.3f}%   (Threshold: {THR_NER}%)")
print(f"  Spike Pct  : {spike_0:.2f}%    (Threshold: {THR_SPIKE}%)")
print(f"  Gauss SNR  : {snr_0:.1f}      (Threshold: {THR_SNR})")

cube = in_cube.copy()

# ── 3. CNN DESTRIPER ──────────────────────────────────────────────────────────
if ner_0 > THR_NER:
    banner("STEP 2  ·  Running CNN Destriper (Spatial Pattern Learning)")
    # Train a 1D CNN on the image rows. It learns to remove periodic column noise.
    # Done band-by-band as stripe patterns differ by detector wavelength.
    
    t0 = time.time()
    for b in range(bands):
        bd = cube[:, :, b].copy()
        if np.isnan(bd).all(): continue
        
        # Fill NaNs temporarily for convolutions
        bd[np.isnan(bd)] = np.nanmedian(bd)
        
        # Prepare tensor (rows, 1, cols)
        X = torch.tensor(bd, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        
        model = CNNDestriper(cols).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # The target is a horizontally smoothed version (truth = no stripes)
        # We teach the network: raw_row -> clean_row
        target = ndimage.uniform_filter1d(bd, size=11, axis=1)
        Y = torch.tensor(target, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        
        for epoch in range(EPOCHS_STR):
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            clean_bd = model(X).squeeze(1).cpu().numpy()
            
        # Put back NaNs where they were
        clean_bd[np.isnan(cube[:, :, b])] = np.nan
        cube[:, :, b] = clean_bd
        
        if b % 10 == 0:
            print(f"    Destriped band {b:02d}/{bands} ...")
            
    print(f"  CNN Destriper completed in {time.time()-t0:.1f}s")
else:
    banner("STEP 2  ·  CNN Destriper SKIPPED (NER is below threshold)")

# ── 4. MAD SPIKE FILTER ───────────────────────────────────────────────────────
if spike_0 > THR_SPIKE:
    banner("STEP 3  ·  Running MAD Spike Filter (Statistical Outlier Removal)")
    t0 = time.time()
    spikes_fixed = 0
    # Median filter along spectral dimension (axis=2)
    # Only replaces pixels that deviate significantly from the median
    for ri in range(rows):
        for ci in range(cols):
            spec = cube[ri, ci, :]
            if not np.any(np.isfinite(spec)): continue
            med = ndimage.median_filter(spec, size=5)
            # Find spikes: divergence from local median
            diff = np.abs(spec - med)
            mad = np.nanmedian(diff) + 1e-6
            z = 0.6745 * diff / mad
            spike_mask = z > 3.5
            if spike_mask.sum() > 0:
                cube[ri, ci, spike_mask] = med[spike_mask]
                spikes_fixed += spike_mask.sum()
                
    print(f"  Spikes detected and fixed: {spikes_fixed:,}")
    print(f"  MAD Filter completed in {time.time()-t0:.1f}s")
else:
    banner("STEP 3  ·  MAD Spike Filter SKIPPED (Spikes below threshold)")

# ── 5. SPECTRAL AUTOENCODER ───────────────────────────────────────────────────
if snr_0 < THR_SNR:
    banner("STEP 4  ·  Running Spectral Autoencoder (Gaussian Denoising)")
    t0 = time.time()
    
    # Flatten cube to list of spectra
    flat_cube = cube.reshape(-1, bands)
    
    # Identify valid pixels (no NaNs)
    valid_mask = np.all(np.isfinite(flat_cube), axis=1)
    train_data = flat_cube[valid_mask]
    n_samples  = train_data.shape[0]
    
    if n_samples > 1000:
        print(f"  Training on {n_samples:,} valid spectra ...")
        # Global scaling for neural network
        spec_min = train_data.min(axis=0, keepdims=True)
        spec_max = train_data.max(axis=0, keepdims=True)
        scale_range = np.clip(spec_max - spec_min, 1e-6, None)
        train_norm = (train_data - spec_min) / scale_range
        
        dataset = TensorDataset(torch.tensor(train_norm, dtype=torch.float32))
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        model = SpectralAutoencoder(bands, hidden=AE_DIM).to(DEVICE)
        
        if os.path.exists(AE_WEIGHTS):
            print(f"  [+] Loading Pre-Trained Autoencoder Weights : {AE_WEIGHTS}")
            model.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
            model.eval()
        else:
            print(f"  [!] No existing weights found. Training new Autoencoder ...")
            optimizer = optim.Adam(model.parameters(), lr=0.005)
            criterion = nn.MSELoss()
            
            for epoch in range(EPOCHS_AE):
                total_loss = 0
                for batch in loader:
                    x = batch[0].to(DEVICE)
                    optimizer.zero_grad()
                    out = model(x)
                    loss = criterion(out, x)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if (epoch+1) % 10 == 0:
                    print(f"    Epoch {epoch+1:02d}/{EPOCHS_AE} — Loss: {total_loss/len(loader):.6f}")
            
            # Save the weights for future scenes
            torch.save(model.state_dict(), AE_WEIGHTS)
            print(f"  [+] Saved new Autoencoder Weights -> {AE_WEIGHTS}")
            model.eval()
                
        # Inference: denoise all valid spectra
        print("  Applying autoencoder to full image ...")
        with torch.no_grad():
            tensor_norm = torch.tensor(train_norm, dtype=torch.float32).to(DEVICE)
            denoised_norm = model(tensor_norm).cpu().numpy()
            
        denoised_spectra = (denoised_norm * scale_range) + spec_min
        
        # Reassemble
        flat_cube[valid_mask] = denoised_spectra
        cube = flat_cube.reshape(rows, cols, bands)
        
    print(f"  Spectral Autoencoder completed in {time.time()-t0:.1f}s")
else:
    banner("STEP 4  ·  Spectral Autoencoder SKIPPED (SNR is above threshold)")

# ── 6. FINAL PROOF & SAVE ─────────────────────────────────────────────────────
banner("STEP 5  ·  Final Noise Measurement & Save")

ner_1, snr_1, spike_1 = measure_noise(cube)

print(f"  ► SNR    : {snr_0:7.1f}   →   {snr_1:7.1f}   (Target: >100)")
print(f"  ► NER    : {ner_0:7.3f}%  →   {ner_1:7.3f}%  (Target: <0.5%)")
print(f"  ► Spikes : {spike_0:7.2f}%  →   {spike_1:7.2f}%  (Target: <1.0%)")

# Save cube
metadata = dict(img.metadata)
metadata['description'] = 'M3 L2: Physics Corrected + ML Denoised (CNN+MAD+Autoencoder)'
spectral.envi.save_image(OUT_HDR, cube.astype(np.float32), metadata=metadata, force=True)
print(f"\n  Saved ML Denoised Cube → {OUT_HDR}")

# Plot Proof Figure
mean_spec_in  = np.nanmean(in_cube.reshape(-1, bands), axis=0)
mean_spec_out = np.nanmean(cube.reshape(-1, bands), axis=0)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0d1117')
for ax in axes:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    ax.xaxis.label.set_color('#8b949e'); ax.yaxis.label.set_color('#8b949e')
    for sp in ax.spines.values(): sp.set_color('#30363d')

ax1, ax2 = axes

ax1.plot(wl, mean_spec_in,  label=f'Input (SNR={snr_0:.1f})', color='#ff7b72', lw=1.5, alpha=0.9)
ax1.plot(wl, mean_spec_out, label=f'Denoised (SNR={snr_1:.1f})', color='#3fb950', lw=1.5, alpha=0.9)
ax1.set_title("Mean Spectrum — Overall Denoising Effect", color='w')
ax1.set_xlabel("Wavelength (nm)"); ax1.set_ylabel("Reflectance")
ax1.legend(facecolor='#0d1117', labelcolor='w')

# Show zoom on a noisy region (e.g. 1500-2000 nm) to prove smoothing
zoom_mask = (wl > 1500) & (wl < 2000)
sample_px = (rows//2, cols//2)

ax2.plot(wl[zoom_mask], in_cube[sample_px[0], sample_px[1], zoom_mask], 
         label='Single Pixel (Input)', color='#ff7b72', lw=1.0)
ax2.plot(wl[zoom_mask], cube[sample_px[0], sample_px[1], zoom_mask], 
         label='Single Pixel (Denoised)', color='#3fb950', lw=1.5)
ax2.set_title(f"Single Pixel Trace ({sample_px[0]},{sample_px[1]}) — Gaussian Smoothing", color='w')
ax2.set_xlabel("Wavelength (nm)")
ax2.legend(facecolor='#0d1117', labelcolor='w')

fig.tight_layout()
fig.savefig(PROOF_FIG, dpi=150, facecolor='#0d1117')
plt.close(fig)
print(f"  Saved Proof Figure → {PROOF_FIG}")
banner("PIPELINE COMPLETE")
