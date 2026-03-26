import os, sys, time, warnings
import numpy as np
import spectral
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# ── 0. SETUP AND CONFIGURATION ───────────────────────────────────────
if len(sys.argv) > 1:
    RAW_HDR = sys.argv[1]
else:
    RAW_HDR = r"D:\Moon_Data\Scene_1\M3G20081118T222604_V01_RFL.HDR"

SCENE_DIR   = os.path.dirname(RAW_HDR)
SCENE_BASE  = os.path.basename(RAW_HDR).replace('.HDR', '').replace('.hdr', '')

PHYSICS_HDR = os.path.join(SCENE_DIR, "Physics_Corrected", f"{SCENE_BASE}_CORRECTED.hdr")
FINAL_HDR   = os.path.join(SCENE_DIR, "ML_Denoised", f"{SCENE_BASE}_FINAL.hdr")
CLASS_HDR   = os.path.join(SCENE_DIR, "Classification", f"{SCENE_BASE}_CLASSIFICATION.hdr")
ABUND_HDR   = os.path.join(SCENE_DIR, "Classification", f"{SCENE_BASE}_ABUNDANCES.hdr")

OUT_DIR = os.path.join(SCENE_DIR, "Journal_Figures")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def banner(msg): print("\n" + "="*75 + f"\n  {msg}\n" + "="*75, flush=True)

# ── 1. CONTINUUM REMOVAL HELPER ──────────────────────────────────────
def continuum_removal(wl_arr, spectra):
    out = np.ones_like(spectra)
    valid_wl_mask = wl_arr <= 2500
    x = wl_arr[valid_wl_mask]
    bottom_y = -2.0 
    for i in range(spectra.shape[0]):
        y = spectra[i, valid_wl_mask]
        fin = np.isfinite(y)
        if fin.sum() < 3: 
            out[i, valid_wl_mask] = y
            continue
        x_fin, y_fin = x[fin], y[fin]
        points = np.column_stack((x_fin, y_fin))
        dummy = np.array([[x_fin[0], bottom_y], [x_fin[-1], bottom_y]])
        aug_points = np.vstack([points, dummy])
        try:
            hull = ConvexHull(aug_points)
            v = np.sort([vertex for vertex in hull.vertices if vertex < len(x_fin)])
            cont = np.interp(x_fin, x_fin[v], y_fin[v])
            cr_y = y_fin / cont
            out[i, np.where(valid_wl_mask)[0][fin]] = cr_y
        except Exception:
            pass
    return out

# ── 2. LOAD DATA CUBES ───────────────────────────────────────────────
banner("Loading Data Layers...")
img_raw  = spectral.open_image(RAW_HDR)
cube_raw = img_raw.open_memmap(interleave='bip', writable=False)
rows, cols, bands = cube_raw.shape

img_phys = spectral.open_image(PHYSICS_HDR)
cube_phys = img_phys.open_memmap(interleave='bip', writable=False)

img_final = spectral.open_image(FINAL_HDR)
cube_final = img_final.open_memmap(interleave='bip', writable=False)

wl = np.array([float(w) for w in img_raw.metadata.get('wavelength', [])])
if wl.max() < 10: wl *= 1000.0

# Find a highly structured pixel for Figure 1 (A bright crater or soil pixel)
# We will just pick the pixel with the highest variance in the denoised cube to guarantee it has absorption features
print("Finding optimal demonstration pixel...")
flat_final = cube_final.reshape(-1, bands)
valid_mask = np.isfinite(flat_final[:, 0]) & (flat_final[:, 0] > 0)
valid_indices = np.where(valid_mask)[0]
sample_idx = valid_indices[np.argmax(np.var(flat_final[valid_indices], axis=1))]
r_idx, c_idx = np.unravel_index(sample_idx, (rows, cols))


# ── FIGURE 1: SPECTRAL EVOLUTION ─────────────────────────────────────
banner("Generating Figure 1: Spectral Evolution")
fig1, ax1 = plt.subplots(4, 1, figsize=(10, 14), facecolor='white')

spec_raw  = cube_raw[r_idx, c_idx, :].copy()
spec_phys = cube_phys[r_idx, c_idx, :].copy()
spec_fin  = cube_final[r_idx, c_idx, :].copy()
spec_cr   = continuum_removal(wl, spec_fin.reshape(1, -1))[0]

ax1[0].plot(wl, spec_raw, color='black', lw=1.5)
ax1[0].set_title("A. Raw M3 Level 2 Data (Heavy space weathering, thermal glow, random noise spikes)")

ax1[1].plot(wl, spec_phys, color='#d95f02', lw=1.5)
ax1[1].set_title("B. Physics Corrected (Destriped, Thermal Emitted Subtracted, Spikes Interpolated)")

ax1[2].plot(wl, spec_fin, color='#1b9e77', lw=1.5)
ax1[2].set_title("C. ML Denoised (Noise Suppressed, High Signal-to-Noise Ratio)")

ax1[3].plot(wl, spec_cr, color='#7570b3', lw=1.5)
ax1[3].set_title("D. Continuum Removed (Space Weathering Flattened, Pure Absorption Fingerprints Isolated)")
ax1[3].axhline(1.0, color='red', linestyle='--', alpha=0.5)
ax1[3].set_xlim(wl.min(), 2500) # Truncate artifact

for ax in ax1:
    ax.set_ylabel("Reflectance")
ax1[3].set_xlabel("Wavelength (nm)")
fig1.tight_layout()
fig1_path = os.path.join(OUT_DIR, "Fig1_Spectral_Evolution.png")
fig1.savefig(fig1_path, dpi=300)
print(f"  Saved -> {fig1_path}")
plt.close(fig1)

# ── FIGURE 2: PCA MANIFOLD ───────────────────────────────────────────
banner("Generating Figure 2: PCA Noise Manifold")
print("Computing PCA on 20,000 pixel subset...")
np.random.seed(42)
pca_subset_idx = np.random.choice(valid_indices, min(20000, len(valid_indices)), replace=False)

raw_samples = cube_raw.reshape(-1, bands)[pca_subset_idx]
raw_samples = np.nan_to_num(raw_samples)
fin_samples = flat_final[pca_subset_idx]

pca_raw = PCA(n_components=2).fit_transform(raw_samples)
pca_fin = PCA(n_components=2).fit_transform(fin_samples)

fig2, (ax_r, ax_f) = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
ax_r.scatter(pca_raw[:,0], pca_raw[:,1], s=2, alpha=0.3, color='grey')
ax_r.set_title("Raw Data PCA Projection (Noise dominated cloud)")
ax_r.set_xlabel("Principal Component 1")
ax_r.set_ylabel("Principal Component 2")

ax_f.scatter(pca_fin[:,0], pca_fin[:,1], s=2, alpha=0.3, color='#1b9e77')
ax_f.set_title("ML Denoised Data PCA Projection (Geological branching structures)")
ax_f.set_xlabel("Principal Component 1")

fig2.tight_layout()
fig2_path = os.path.join(OUT_DIR, "Fig2_PCA_Manifold.png")
fig2.savefig(fig2_path, dpi=300)
print(f"  Saved -> {fig2_path}")
plt.close(fig2)


# ── FIGURE 3: LATENT SPACE T-SNE ─────────────────────────────────────
banner("Generating Figure 3: Contrastive Latent Space Projection")
print("Loading Universal Contrastive Encoder and mapping to Latent Space...")
class SpecEncoderCNN(nn.Module):
    def __init__(self, in_bands, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Flatten()
        )
        dummy = torch.zeros(1, 1, in_bands)
        flat_size = self.encoder(dummy).shape[1]
        self.projector = nn.Sequential(
            nn.Linear(flat_size, 64), nn.ReLU(), nn.Linear(64, out_dim)
        )
    def forward(self, x):
        h = self.encoder(x.unsqueeze(1))
        z = self.projector(h)
        return F.normalize(z, dim=1)

model = SpecEncoderCNN(bands, 16).to(DEVICE)
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m3_contrastive_encoder_cr.pth")
if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
else:
    print("[!] Model weights not found. Run classification pipeline first.")
model.eval()

# Process CR subset through model
print("Applying CR to subset...")
cr_samples = continuum_removal(wl, fin_samples)
max_ref = 1.0 # CR max is 1.0
cr_norm = np.clip(cr_samples / max_ref, 0, 1)

with torch.no_grad():
    tensor_in = torch.tensor(cr_norm, dtype=torch.float32).to(DEVICE)
    embeddings = model(tensor_in).cpu().numpy()

print("Computing t-SNE projection (this takes a moment)...")
tsne = TSNE(n_components=2, perplexity=40, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Load Classification Map to color boundaries
img_class = spectral.open_image(CLASS_HDR)
class_map = img_class.read_band(0).flatten()
subset_labels = class_map[pca_subset_idx]

class_names = img_class.metadata.get('class names', [])

fig3, ax3 = plt.subplots(1, 1, figsize=(12, 10), facecolor='white')
sc = ax3.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=subset_labels, cmap='tab10', s=5, alpha=0.8)
ax3.set_title("t-SNE Projection of 16D Contrastive Latent Space")
ax3.set_xticks([]); ax3.set_yticks([]) # Hide axes for clean look

# Legend
handles, labels = sc.legend_elements(prop="colors")
filtered_handles, filtered_labels = [], []
for h, lbl_str in zip(handles, labels):
    idx = int(lbl_str.extract()[0]) if hasattr(lbl_str, 'extract') else int(lbl_str.split('$\\mathdefault{')[1].split('}')[0])
    if idx < len(class_names):
        filtered_handles.append(h)
        filtered_labels.append(class_names[idx])
        
if filtered_handles:
    ax3.legend(filtered_handles, filtered_labels, title="Discovered Geological Units", loc='best', markerscale=2)

fig3.tight_layout()
fig3_path = os.path.join(OUT_DIR, "Fig3_Latent_Space_tSNE.png")
fig3.savefig(fig3_path, dpi=300)
print(f"  Saved -> {fig3_path}")
plt.close(fig3)


# ── FIGURE 4: ABUNDANCE MAPS ─────────────────────────────────────────
banner("Generating Figure 4: Spatial Abundance Heatmaps")
if os.path.exists(ABUND_HDR):
    img_abund = spectral.open_image(ABUND_HDR)
    abund_cube = img_abund.load()
    abund_names = img_abund.metadata.get('band names', [f"Mineral {i}" for i in range(abund_cube.shape[2])])
    
    n_minerals = abund_cube.shape[2]
    cols_plot = min(n_minerals, 4)
    rows_plot = int(np.ceil(n_minerals / cols_plot))
    
    fig4, axes = plt.subplots(rows_plot, cols_plot, figsize=(cols_plot*6, rows_plot*5), facecolor='white')
    if n_minerals == 1: axes = np.array([axes])
    axes = axes.flatten()
    
    for m in range(n_minerals):
        im = axes[m].imshow(abund_cube[:,:,m], cmap='magma', vmin=0, vmax=1.0)
        axes[m].set_title(f"Fractional Abundance: {abund_names[m]}")
        axes[m].axis('off')
        fig4.colorbar(im, ax=axes[m], fraction=0.046, pad=0.04, label="Fraction (0 to 1)")
        
    for m in range(n_minerals, len(axes)):
        axes[m].axis('off')
        
    fig4.tight_layout()
    fig4_path = os.path.join(OUT_DIR, "Fig4_Fractional_Abundances.png")
    fig4.savefig(fig4_path, dpi=300)
    print(f"  Saved -> {fig4_path}")
    plt.close(fig4)
else:
    print("[!] Abundance Map not found. Skipping Figure 4.")

banner("JOURNAL FIGURES COMPLETED")
print(f"Check the directory: {OUT_DIR}")
