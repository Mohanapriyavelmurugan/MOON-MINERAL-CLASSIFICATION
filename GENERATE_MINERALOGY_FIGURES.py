import os, sys, time, warnings
import numpy as np
import spectral
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score
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

FINAL_HDR   = os.path.join(SCENE_DIR, "ML_Denoised", f"{SCENE_BASE}_FINAL.hdr")
CLASS_HDR   = os.path.join(SCENE_DIR, "Classification", f"{SCENE_BASE}_CLASSIFICATION.hdr")
ABUND_HDR   = os.path.join(SCENE_DIR, "Classification", f"{SCENE_BASE}_ABUNDANCES.hdr")
USGS_DIR    = r"D:\spectral_libraries\USGS"
RELAB_DIR   = r"D:\spectral_libraries\RELAB"

OUT_DIR = os.path.join(SCENE_DIR, "Mineralogy_Figures")
os.makedirs(OUT_DIR, exist_ok=True)

try:
    plt.style.use('ggplot')
except Exception:
    pass

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

# ── 2. LOAD DATA ─────────────────────────────────────────────────────
banner("Loading Data Layers for Mineralogy Validation...")
img_final = spectral.open_image(FINAL_HDR)
cube_final = img_final.open_memmap(interleave='bip', writable=False)
rows, cols, bands = cube_final.shape
wl = np.array([float(w) for w in img_final.metadata.get('wavelength', [])])
if wl.max() < 10: wl *= 1000.0

img_class = spectral.open_image(CLASS_HDR)
class_map = img_class.read_band(0)
class_names = img_class.metadata.get('class names', [])

flat_final = cube_final.reshape(-1, bands)
valid_mask = np.isfinite(flat_final[:, 0]) & (flat_final[:, 0] > 0)
valid_indices = np.where(valid_mask)[0]
N_VALID = len(valid_indices)

# ── VISUAL A: AUTO-K DB DISCOVERY CURVE ──────────────────────────────
banner("Visual A: Auto-K DB Curve (Computing on Subset)...")
np.random.seed(42)
k_subset_idx = np.random.choice(valid_indices, min(15000, N_VALID), replace=False)
k_samples = flat_final[k_subset_idx]
print("Applying CR to K-samples...")
k_cr = continuum_removal(wl, k_samples)
k_cr_norm = np.clip(k_cr / 1.0, 0, 1)

cluster_range = range(2, 9)
db_scores = []
for k in cluster_range:
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=3)
    labels = kmeans.fit_predict(k_cr_norm)
    score = davies_bouldin_score(k_cr_norm, labels)
    db_scores.append(score)

optimal_k = cluster_range[np.argmin(db_scores)]

figA, axA = plt.subplots(figsize=(8, 6), facecolor='white')
axA.plot(cluster_range, db_scores, marker='o', linestyle='-', color='purple', linewidth=2)
axA.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
axA.set_title("Visual A: Davies-Bouldin Auto-K Discovery (Mineral Clusters)")
axA.set_xlabel("Number of Assumed Mineral Units (K)")
axA.set_ylabel("Davies-Bouldin Score (Lower = Better Chemistry Separation)")
axA.legend()
figA.savefig(os.path.join(OUT_DIR, "VisA_AutoK_DBCurve.png"), dpi=300)
plt.close(figA)


# ── VISUAL B: LUNAR ENDMEMBER BUNDLE ─────────────────────────────────
banner("Visual B: Extracting Lunar Endmember Bundle...")
class_map_flat = class_map.flatten()
n_classes = len(class_names) - 1 # exclude Unclassified (0)

cluster_centers_raw = []
for i in range(1, n_classes + 1):
    c_mask = (class_map_flat[valid_indices] == i)
    if np.sum(c_mask) > 0:
        mean_spec = np.mean(flat_final[valid_indices[c_mask]], axis=0)
        cluster_centers_raw.append(mean_spec)

cluster_centers_raw = np.array(cluster_centers_raw)
cluster_centers_cr = continuum_removal(wl, cluster_centers_raw)

figB, axB = plt.subplots(figsize=(10, 6), facecolor='white')
colors = plt.get_cmap('tab10').colors
for i in range(len(cluster_centers_cr)):
    axB.plot(wl, cluster_centers_cr[i], color=colors[i % 10], linewidth=2, label=class_names[i+1])

axB.axhline(1.0, color='black', linestyle='--')
axB.set_xlim(wl.min(), 2500)
axB.set_title("Visual B: Unified Lunar Endmember Bundle (Continuum Removed)")
axB.set_xlabel("Wavelength (nm)")
axB.set_ylabel("Normalized Reflectance Ratio")
axB.legend(loc='lower right', fontsize=8)
figB.savefig(os.path.join(OUT_DIR, "VisB_Endmember_Bundle.png"), dpi=300)
plt.close(figB)


# ── VISUAL C: SPECTRAL SEMANTIC MATCH MATRIX ─────────────────────────
banner("Visual C: Semantic Match Grid (Library Search)...")

def load_libraries():
    import glob
    lib_spectra, lib_names = [], []
    # 1. USGS
    wl_file = os.path.join(USGS_DIR, "ASCIIdata_splib07b_cvM3-target", "s07_M3t_M3_Wavelengths_TARGET_2011t3_micron_256c.txt")
    if os.path.exists(wl_file):
        with open(wl_file, 'r') as f: lines = f.readlines()
        usgs_wls = np.array([float(line.split()[0])*1000.0 for line in lines[1:] if line.split()])
        
        cvM3_dir = os.path.join(USGS_DIR, "ASCIIdata_splib07b_cvM3-target")
        for path in glob.glob(os.path.join(cvM3_dir, '**', 's07_M3t_*.txt'), recursive=True):
            if any(k in os.path.basename(path).lower() for k in ['pyroxene', 'olivine', 'plagioclase', 'anorthosite', 'glass', 'augite', 'diopside', 'enstatite', 'bronzite']):
                with open(path, 'r') as file:
                    refs = [float(line.split()[0]) for line in file.readlines()[1:] if line.split()]
                    refs_arr = np.array(refs)
                    valid = ~np.isnan(refs_arr) & (refs_arr >= 0)
                    if np.sum(valid) > 10:
                        lib_names.append(f"USGS: {os.path.basename(path).replace('s07_M3t_', '').replace('.txt', '')}")
                        lib_spectra.append(np.interp(wl, usgs_wls[valid], refs_arr[valid]))

    # 2. RELAB
    if os.path.exists(RELAB_DIR):
        for f in os.listdir(RELAB_DIR):
            if f.endswith('.tab'):
                with open(os.path.join(RELAB_DIR, f), 'r') as file:
                    try:
                        wls, refs = [], []
                        for line in file.readlines()[1:]:
                            parts = line.split()
                            if len(parts) >= 2:
                                try: wls.append(float(parts[0])); refs.append(float(parts[1]))
                                except ValueError: break
                        if len(wls) > 10:
                            uwls, indices = np.unique(wls, return_index=True)
                            lib_names.append(f"RELAB: {f.replace('.tab','')}")
                            lib_spectra.append(np.interp(wl, uwls, np.array(refs)[indices]))
                    except Exception: pass
    return lib_names, np.array(lib_spectra)

lib_names, lib_interp = load_libraries()

if len(lib_interp) == 0:
    print("  [!] Failed to load ASCII Earth libraries. Skipping Visual C.")
else:
    lib_cr = continuum_removal(wl, lib_interp)
    lib_cr_norm = np.clip(lib_cr / 1.0, 0, 1)
cc_cr_norm = np.clip(cluster_centers_cr / 1.0, 0, 1)

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
model.eval()

with torch.no_grad():
    lib_tensor = torch.tensor(lib_cr_norm, dtype=torch.float32).to(DEVICE)
    cc_tensor = torch.tensor(cc_cr_norm, dtype=torch.float32).to(DEVICE)
    lib_embed = model(lib_tensor)
    cc_embed = model(cc_tensor)
    
    sim_matrix = torch.mm(cc_embed, lib_embed.t()).cpu().numpy()

n_plots = len(cc_cr_norm)
cols_plot = min(n_plots, 2)
rows_plot = int(np.ceil(n_plots / cols_plot))

figC, axesC = plt.subplots(rows_plot, cols_plot, figsize=(12, rows_plot*5), facecolor='white')
if n_plots == 1: axesC = np.array([axesC])
axesC_flat = axesC.flatten()

for i in range(n_plots):
    best_match_idx = np.argmax(sim_matrix[i])
    best_score = sim_matrix[i, best_match_idx]
    
    axesC_flat[i].plot(wl, cc_cr_norm[i], color=colors[i % 10], linewidth=3, label=f'Lunar Cluster {i+1}')
    axesC_flat[i].plot(wl, lib_cr_norm[best_match_idx], color='black', linestyle='--', linewidth=2, label=f'Library Match: {lib_names[best_match_idx]}')
    
    axesC_flat[i].set_title(f"Match Confidence: {best_score*100:.1f}%")
    axesC_flat[i].set_xlim(wl.min(), 2500)
    axesC_flat[i].axhline(1.0, color='gray', linestyle=':')
    axesC_flat[i].set_xlabel("Wavelength (nm)")
    axesC_flat[i].legend(fontsize=8)

for i in range(n_plots, len(axesC_flat)):
    axesC_flat[i].axis('off')
    
figC.tight_layout()
figC.savefig(os.path.join(OUT_DIR, "VisC_Contrastive_Library_Overlays.png"), dpi=300)
plt.close(figC)


# ── VISUAL D & E: NNLS UNMIXING & RMSE ───────────────────────────────
banner("Visual D/E: Sub-pixel Abundances and RMSE Residuals...")
img_abund = spectral.open_image(ABUND_HDR)
abund_cube = img_abund.load()
abund_names = img_abund.metadata.get('band names', [f"Mineral {i}" for i in range(abund_cube.shape[2])])

# Vis D: Individual Grids
n_minerals = abund_cube.shape[2]
figD, axesD = plt.subplots(int(np.ceil(n_minerals / 2)), 2, figsize=(14, n_minerals*3), facecolor='white')
if n_minerals <= 2: axesD = np.array([axesD]).flatten()
else: axesD = axesD.flatten()

for m in range(n_minerals):
    im = axesD[m].imshow(abund_cube[:,:,m], cmap='magma', vmin=0, vmax=1.0)
    axesD[m].set_title(f"Visual D: Unmixed Geo-Map ({abund_names[m]})")
    axesD[m].axis('off')
    figD.colorbar(im, ax=axesD[m], fraction=0.046, pad=0.04)
for m in range(n_minerals, len(axesD)): axesD[m].axis('off')

figD.tight_layout()
figD.savefig(os.path.join(OUT_DIR, "VisD_Individual_Mineral_Geomaps.png"), dpi=300)
plt.close(figD)

# Vis E: NNLS RMSE Error Map
abund_flat = abund_cube.reshape(-1, n_minerals)

# Reconstruct RAW spectrum from abundances * Earth Library Endmembers
E = []
for name in abund_names:
    if name in lib_names:
        idx = lib_names.index(name)
        E.append(lib_interp[idx])
    else:
        E.append(np.zeros(bands))
E = np.array(E) # Shape (n_minerals, bands)

reconstructed_flat = np.dot(abund_flat, E)
residuals = flat_final - reconstructed_flat

# Mask out background to prevent skewing RMSE colors
residuals[~valid_mask] = 0.0

rmse_flat = np.sqrt(np.mean(residuals**2, axis=-1))
rmse_map = rmse_flat.reshape(rows, cols)

figE, axE = plt.subplots(figsize=(8, 12), facecolor='white')
# Cap the visualization at 95th percentile to highlight true errors
vmax = np.percentile(rmse_flat[valid_mask], 95)
imE = axE.imshow(rmse_map, cmap='Reds', vmin=0, vmax=vmax)
axE.set_title("Visual E: Sub-Pixel Unmixing Residual Error (RMSE)")
axE.axis('off')
figE.colorbar(imE, ax=axE, shrink=0.8, label="Root Mean Square Error")
figE.savefig(os.path.join(OUT_DIR, "VisE_NNLS_RMSE_ErrorMap.png"), dpi=300)
plt.close(figE)

banner("MINERALOGY FIGURES COMPLETED")
print(f"Check the directory: {OUT_DIR}")
