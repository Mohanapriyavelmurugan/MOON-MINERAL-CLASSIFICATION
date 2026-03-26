import os, sys, time, warnings
import numpy as np
import random
import spectral
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score
from scipy.optimize import nnls
from scipy.spatial import ConvexHull

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# ── CONFIGURATION ────────────────────────────────────
if len(sys.argv) > 1:
    RAW_HDR = sys.argv[1]
else:
    RAW_HDR = r"D:\Moon_Data\Scene_2\M3G20081201T064047_V01_RFL.HDR"

SCENE_DIR   = os.path.dirname(RAW_HDR)
SCENE_BASE  = os.path.basename(RAW_HDR).replace('.HDR', '').replace('.hdr', '')
IN_HDR      = os.path.join(SCENE_DIR, "ML_Denoised", f"{SCENE_BASE}_FINAL.hdr")
OUT_DIR     = os.path.join(SCENE_DIR, "Classification")
CLASS_HDR   = os.path.join(OUT_DIR, f"{SCENE_BASE}_CLASSIFICATION.hdr")
CONF_HDR    = os.path.join(OUT_DIR, f"{SCENE_BASE}_CONFIDENCE.hdr")
ABUND_HDR   = os.path.join(OUT_DIR, f"{SCENE_BASE}_ABUNDANCES.hdr")
PROOF_FIG   = os.path.join(OUT_DIR, "contrastive_matches_proof.png")

RELAB_DIR   = r"D:\spectral_libraries\RELAB"

# ML Hyperparameters
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS      = 5
BATCH_SIZE  = 1024
EMBED_DIM   = 16
TEMPERATURE = 0.1

os.makedirs(OUT_DIR, exist_ok=True)

def banner(msg): print("\n" + "="*75 + f"\n  {msg}\n" + "="*75, flush=True)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
banner("STEP 1  ·  Loading Denoised Cube")
if not os.path.exists(IN_HDR):
    print(f"[!] Input file not found: {IN_HDR}")
    print("Run `MOON_ML_DENOISING.py` first.")
    sys.exit(1)

img = spectral.open_image(IN_HDR)
# Use memory mapping to prevent Out-Of-Memory crashes on 8.5M+ pixel scenes
cube = img.open_memmap(interleave='bip', writable=False)
rows, cols, bands = cube.shape
print(f"  Shape: {rows} × {cols} × {bands}")

wl = np.array([float(w) for w in img.metadata.get('wavelength', [])])
if wl.max() < 10: wl *= 1000.0

# Extract valid pixels
print("  Scanning for valid pixels (ignoring background space) ...")
flat_cube = cube.reshape(-1, bands)
# Create a valid mask band-by-band to save memory
valid_counts = np.zeros(rows * cols, dtype=np.int32)
for b in range(bands):
    valid_counts += np.isfinite(flat_cube[:, b]).astype(np.int32)
valid_mask = valid_counts > (bands * 0.8) # 80% valid

valid_pixels = np.nan_to_num(flat_cube[valid_mask], nan=0.0)
n_samples = valid_pixels.shape[0]
print(f"  Valid pixels for clustering: {n_samples:,} / {rows*cols:,}")

if n_samples == 0:
    print("[!] No valid pixels found. Exiting.")
    sys.exit(1)

def continuum_removal(wl_arr, spectra):
    """
    Applies convex-hull continuum removal strictly up to 2500 nm.
    spectra: (N, bands) array
    Returns array where values are Ref / Continuum.
    For wl > 2500, forces values to 1.0 to cleanly truncate the artifact tail
    without changing the 85-band matrix dimension shape required by PyTorch.
    """
    out = np.ones_like(spectra)
    valid_wl_mask = wl_arr <= 2500
    x = wl_arr[valid_wl_mask]
    
    # Add dummy points at the bottom to force the upper convex hull envelope
    bottom_y = -2.0 
    
    for i in range(spectra.shape[0]):
        y = spectra[i, valid_wl_mask]
        fin = np.isfinite(y)
        if fin.sum() < 3: 
            out[i, valid_wl_mask] = y
            continue
            
        x_fin = x[fin]
        y_fin = y[fin]
        
        points = np.column_stack((x_fin, y_fin))
        dummy = np.array([[x_fin[0], bottom_y], [x_fin[-1], bottom_y]])
        aug_points = np.vstack([points, dummy])
        
        try:
            hull = ConvexHull(aug_points)
            v = np.sort([vertex for vertex in hull.vertices if vertex < len(x_fin)])
            continuum = np.interp(x_fin, x_fin[v], y_fin[v])
            cr_y = y_fin / continuum
            out[i, np.where(valid_wl_mask)[0][fin]] = cr_y
        except Exception:
            pass
            
    return out

t0_cr = time.time()
print("  Applying Convex Hull Continuum Removal to M3 pixels (Truncating > 2500 nm) ...")
valid_pixels = continuum_removal(wl, valid_pixels)
print(f"  Continuum Removal complete in {time.time()-t0_cr:.1f}s")

# Normalize Data for Network
max_ref = np.percentile(valid_pixels, 99)
norm_pixels = np.clip(valid_pixels / max_ref, 0, 1)

# ── 2. CONTRASTIVE DATASET & NETWORK ──────────────────────────────────────────
class ContrastiveDataset(Dataset):
    def __init__(self, data):
        # Push all 1.8M lightweight spectra instantly to GPU VRAM (~600 MB)
        # This completely removes the severe CPU-to-GPU bottleneck during augmentation
        self.data = torch.tensor(data, dtype=torch.float32).to(DEVICE)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        x = self.data[idx]
        
        # Augmentation 1: Gaussian noise + small scaling (happens instantly on GPU)
        noise1 = torch.randn_like(x) * 0.01
        scale1 = 1.0 + (torch.rand(1, device=DEVICE).item() - 0.5) * 0.05
        aug1 = torch.clamp((x * scale1) + noise1, 0, 1)
        
        # Augmentation 2: Band Dropout
        aug2 = x.clone()
        drop_mask = torch.rand_like(x) < 0.1 # 10% dropout
        aug2[drop_mask] = 0.0
        
        return x, aug1, aug2

class SpecEncoderCNN(nn.Module):
    def __init__(self, in_bands, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute flatten size dynamically
        dummy = torch.zeros(1, 1, in_bands)
        flat_size = self.encoder(dummy).shape[1]
        
        self.projector = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1) # (B, 1, Bands)
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1) # L2 normalize embedding

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0) # 2N x D
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature
        sim_matrix.fill_diagonal_(-9e15) # mask self
        
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0).to(z.device)
        return self.criterion(sim_matrix, labels)

# ── 3. PRE-TRAINING ──────────────────────────────────────────────────────────
banner("STEP 2  ·  Training Contrastive SpecEncoder")
dataset = ContrastiveDataset(norm_pixels)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SpecEncoderCNN(bands, EMBED_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = NTXentLoss(temperature=TEMPERATURE)

# Foundation Model: Check for pre-trained weights
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m3_contrastive_encoder_cr.pth")

if os.path.exists(WEIGHTS_PATH):
    print(f"  [+] Loading Pre-Trained Universal Contrastive Encoder : {WEIGHTS_PATH}", flush=True)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
else:
    print(f"  [!] No existing weights found. Training entirely new Universal Contrastive Encoder ...", flush=True)
    t0 = time.time()
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for original, aug1, aug2 in loader:
            aug1 = aug1.to(DEVICE)
            aug2 = aug2.to(DEVICE)
            
            optimizer.zero_grad()
            z1 = model(aug1)
            z2 = model(aug2)
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{EPOCHS} — NT-Xent Loss: {total_loss/len(loader):.4f}", flush=True)
    
    # Save the Foundation Model weights for all future scenes
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"  [+] Saved new Universal Contrastive Weights -> {WEIGHTS_PATH}", flush=True)
    print(f"  Training Time: {time.time()-t0:.1f}s", flush=True)

# ── 4. EMBEDDING EXTRACTION ───────────────────────────────────────────────────
banner("STEP 3  ·  Extracting Latent Embeddings & Clustering")
model.eval()
t0 = time.time()
embeddings_list = []
eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE*2, shuffle=False)

with torch.no_grad():
    for x, _, _ in eval_loader:
        x = x.to(DEVICE)
        z = model(x)
        embeddings_list.append(z.cpu().numpy())
embeddings = np.vstack(embeddings_list)
print(f"  Embeddings extracted: {embeddings.shape[0]} × {embeddings.shape[1]}")

# ── Auto-K Discovery (Silhouette/Davies-Bouldin) 
print(f"  Running Auto-K Discovery (K=3 to 10)...")
best_k = 4
best_score = float('inf')

# Sample for fast scoring
sample_size = min(50000, embeddings.shape[0])
idx = np.random.choice(embeddings.shape[0], sample_size, replace=False)
emb_sample = embeddings[idx]

for k in range(3, 11):
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096)
    labels = km.fit_predict(emb_sample)
    if len(set(labels)) > 1:
        score = davies_bouldin_score(emb_sample, labels)
        print(f"    Tested K={k}: Davies-Bouldin Score = {score:.3f}")
        if score < best_score:
            best_score = score
            best_k = k

N_CLUSTERS = best_k
print(f"  [+] Optimal Spectral Clusters Discovered: K={N_CLUSTERS}")

print(f"  Running Final Clustering (K={N_CLUSTERS})...")
kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, batch_size=4096)
cluster_labels = kmeans.fit_predict(embeddings)
centroids = kmeans.cluster_centers_

# Confidence Calculation (Cosine Similarity to Centroid)
# Embeddings and centroids are L2 normalized, so dot product = cosine similarity
norm_cen = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
confidences = np.sum(embeddings * norm_cen[cluster_labels], axis=1) # Shape: (N,)
confidences = np.clip(confidences, 0.0, 1.0) * 100.0 # to percentage
print(f"  Clustering total time: {time.time()-t0:.1f}s")


# ── 5. LIBRARY MATCHING (RELAB) ───────────────────────────────────────────────
banner("STEP 4  ·  Spectral Library Matching (Naming the discovered clusters)")

def load_relab_library(lib_dir):
    library = {}
    if not os.path.exists(lib_dir):
        print(f"  [!] RELAB dir not found: {lib_dir}")
        return library
        
    for f in os.listdir(lib_dir):
        if not f.endswith('.tab'): continue
        path = os.path.join(lib_dir, f)
        try:
            with open(path, 'r') as file:
                lines = file.readlines()
            wls, refs = [], []
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        wl_val, ref_val = float(parts[0]), float(parts[1])
                        wls.append(wl_val)
                        refs.append(ref_val)
                    except ValueError:
                        break # End of data / start of metadata
            if len(wls) > 10:
               wls, indices = np.unique(wls, return_index=True)
               refs = np.array(refs)[indices]
               library[f"RELAB: {f.replace('.tab','')}"] = (wls, refs)
        except Exception:
            pass
    return library

def load_usgs_library(usgs_dir):
    library = {}
    import glob
    
    # Path to wavelengths
    wl_file = os.path.join(usgs_dir, "ASCIIdata_splib07b_cvM3-target", "s07_M3t_M3_Wavelengths_TARGET_2011t3_micron_256c.txt")
    if not os.path.exists(wl_file):
        print(f"  [!] USGS wavelengths file not found at {wl_file}")
        return library
        
    try:
        with open(wl_file, 'r') as f:
            lines = f.readlines()
        
        # Wavelengths are from line 2 onwards, 1 value per line, in microns.
        # We need to convert them to nanometers (x1000)
        usgs_wls = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                usgs_wls.append(float(parts[0]) * 1000.0)
        usgs_wls = np.array(usgs_wls)
    except Exception as e:
        print(f"  [!] Failed to parse USGS wavelengths: {e}")
        return library

    cvM3_dir = os.path.join(usgs_dir, "ASCIIdata_splib07b_cvM3-target")
    txt_files = glob.glob(os.path.join(cvM3_dir, '**', 's07_M3t_*.txt'), recursive=True)
    
    keywords = ['pyroxene', 'olivine', 'plagioclase', 'anorthosite', 'ilmenite', 
                'spinel', 'glass', 'augite', 'diopside', 'enstatite', 'pigeonite', 
                'bronzite', 'bytownite', 'labradorite', 'troctolite', 'norite']
    
    for path in txt_files:
        basename = os.path.basename(path).lower()
        if not any(k in basename for k in keywords):
            continue
            
        try:
            with open(path, 'r') as file:
                lines = file.readlines()
            refs = []
            # Reflectance from line 2
            for line in lines[1:]:
                parts = line.split()
                if parts:
                    val = float(parts[0])
                    # Handle USGS missing values
                    if val < -1e10: val = np.nan
                    refs.append(val)
                    
            if len(refs) == len(usgs_wls):
                refs_arr = np.array(refs)
                valid = ~np.isnan(refs_arr) & (refs_arr >= 0)
                if np.sum(valid) > 10:
                    name_clean = os.path.basename(path).replace("s07_M3t_", "").replace(".txt", "")
                    library[f"USGS: {name_clean}"] = (usgs_wls[valid], refs_arr[valid])
        except Exception:
            pass
            
    return library

# Calculate Mean Spectrum of each Cluster (from original data, not embeddings)
cluster_means = np.zeros((N_CLUSTERS, bands), dtype=np.float32)
for k in range(N_CLUSTERS):
    mask = (cluster_labels == k)
    if mask.sum() > 0:
        cluster_means[k] = np.mean(valid_pixels[mask], axis=0)

# Load and Interpolate Both Libraries
relab_lib = load_relab_library(RELAB_DIR)
print(f"  Loaded {len(relab_lib)} lunar spectra from RELAB.")

usgs_dir = r"D:\spectral_libraries\USGS\usgs_splib07\ASCIIdata"
usgs_lib = load_usgs_library(usgs_dir)
print(f"  Loaded {len(usgs_lib)} mineral spectra from USGS.")

# Master library combined
master_lib_raw = {**relab_lib, **usgs_lib}
master_lib = {}
print(f"  Total library size: {len(master_lib_raw)} pristine spectra combined.")
print("  Interpolating and applying Continuum Removal to Library Spectra ...")
for name, (l_wl, l_ref) in master_lib_raw.items():
    interp_func = interp1d(l_wl, l_ref, kind='linear', bounds_error=False, fill_value='extrapolate')
    l_spec_interp = interp_func(wl)
    # Apply CR exactly like M3
    l_spec_cr = continuum_removal(wl, l_spec_interp.reshape(1, -1))[0]
    master_lib[name] = (wl, l_spec_cr)

# Match each cluster
cluster_names = {}
cluster_colors = {}
cmap = plt.get_cmap('tab10' if N_CLUSTERS <= 10 else 'tab20')

print("  Encoding library spectra into Latent Space for semantic matching...")
library_embeddings = {}
model.eval()
with torch.no_grad():
    for name, (l_wl, l_ref_cr) in master_lib.items():
        # No interpolation needed here, already performed above
        l_norm = np.clip(l_ref_cr, 0, 1)
        
        # Pass to Neural Network
        l_tensor = torch.tensor(l_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        z_lib = model(l_tensor)
        library_embeddings[name] = z_lib.cpu().numpy()[0] # 16D normalized latent vector

# Match each cluster centroid embedding vs library embeddings using Cosine Similarity
cluster_names = {}
cluster_colors = {}
cmap = plt.get_cmap('tab10' if N_CLUSTERS <= 10 else 'tab20')

for k in range(N_CLUSTERS):
    c_emb = norm_cen[k]  # The L2 normalized latent centroid
    best_match_name = f"Unclassified_Semantic_Unit_{k+1}"
    best_sim = -1.0
    
    for name, z_lib in library_embeddings.items():
        # Cosine similarity is dot product of L2 normalized vectors
        sim = float(np.dot(c_emb, z_lib))
        if sim > best_sim:
            best_sim = sim
            best_match_name = name
    
    # Accept anything strongly correlated in the semantic space (latent space separates noise naturally)
    MIN_LATENT_SIM = 0.50
    if best_sim < MIN_LATENT_SIM:
        cluster_names[k] = f"Unclassified_Semantic_Unit_{k+1}"
        print(f"  Cluster {k+1}  -->  UNCLASSIFIED  (best latent sim: {best_sim:.3f} < {MIN_LATENT_SIM})")
    else:
        cluster_names[k] = best_match_name
        print(f"  Cluster {k+1}  -->  {best_match_name}  (Latent Sim: {best_sim:.3f})")
    cluster_colors[k] = cmap(k)[:3]

# ── Summary: all unique minerals discovered in this scene
unique_minerals = sorted(set(cluster_names[k] for k in range(N_CLUSTERS)
                             if not cluster_names[k].startswith("Unclassified")))
print(f"\n  ╔══ Minerals identified in this scene ({len(unique_minerals)}) ══")
for m in unique_minerals:
    matched_clusters = [k+1 for k in range(N_CLUSTERS) if cluster_names[k] == m]
    print(f"  ║  {m}  →  clusters {matched_clusters}")
if not unique_minerals:
    print("  ║  No minerals passed the similarity threshold — consider lowering MIN_PEARSON.")
print("  ╚" + "═"*45)

# ── 6. SUBPIXEL FRACTIONAL UNMIXING ───────────────────────────────────────────
banner("STEP 5  ·  Subpixel Fractional Unmixing")

print("  Unmixing 1.8M pixels against latent endmembers...")
t0_unmix = time.time()
endmember_names = [cluster_names[k] for k in range(N_CLUSTERS) if not cluster_names[k].startswith("Unclassified")]
endmember_names = list(set(endmember_names)) # unique endmembers
n_endmembers = len(endmember_names)

fraction_map_flat = np.zeros((rows * cols, max(n_endmembers, 1)), dtype=np.float32)

if n_endmembers > 0:
    # Build Endmember Matrix E (Bands x E) from Library
    E = np.zeros((bands, n_endmembers), dtype=np.float32)
    for i, name in enumerate(endmember_names):
        l_wl, l_ref = master_lib[name]
        interp_func = interp1d(l_wl, l_ref, kind='linear', bounds_error=False, fill_value='extrapolate')
        E[:, i] = interp_func(wl)
    
    # Vectorized Least Squares (fast NNLS approximation)
    print("  Running Matrix Least Squares on valid pixels (chunked to save RAM) ...")
    # Resolve memory spike by chunking the pseudo-inverse solver
    fractions = np.zeros((n_samples, n_endmembers), dtype=np.float32)
    chunk_size = 50000
    for i in range(0, n_samples, chunk_size):
        chunk = valid_pixels[i:i+chunk_size]
        f_T_chunk, _, _, _ = np.linalg.lstsq(E, chunk.T, rcond=None)
        fractions[i:i+chunk_size] = f_T_chunk.T # shape (chunk_size, E)
    
    # Apply Non-Negative constraint (clip) and sum-to-1 normalization (~100% reflectance components)
    fractions = np.clip(fractions, 0, None)
    sum_f = np.sum(fractions, axis=1, keepdims=True)
    fractions = np.where(sum_f > 1e-5, fractions / sum_f, 0)
    
    # Place valid fractions back into the flattened 2D spatial image
    fraction_map_flat[valid_mask.flatten()] = fractions
    abundance_cube = fraction_map_flat.reshape((rows, cols, n_endmembers))
    print(f"  Unmixing completed in {time.time()-t0_unmix:.1f}s")
else:
    abundance_cube = np.zeros((rows, cols, 1), dtype=np.float32)
    print("  No classified endmembers to unmix.")

# ── 7. REASSEMBLE AND SAVE ────────────────────────────────────────────────────
banner("STEP 6  ·  Writing Final Maps")

full_class_map = np.zeros((rows, cols), dtype=np.uint8)
full_conf_map  = np.zeros((rows, cols), dtype=np.float32)

full_class_map_flat = full_class_map.flatten()
full_conf_map_flat  = full_conf_map.flatten()

# We map k+1 so 0 is background
full_class_map_flat[valid_mask] = (cluster_labels + 1).astype(np.uint8)
full_conf_map_flat[valid_mask]  = confidences

full_class_map = full_class_map_flat.reshape(rows, cols)
full_conf_map = full_conf_map_flat.reshape(rows, cols)

# Save Maps
# Create class names for ENVI header
class_names_list = ["Unclassified"] + [cluster_names[k] for k in range(N_CLUSTERS)]
class_colors_list = [[0,0,0]] + [[int(c*255) for c in cluster_colors[k]] for k in range(N_CLUSTERS)]

meta_class = dict(img.metadata)
meta_class['bands'] = 1
meta_class['data type'] = 1 # Byte
meta_class['classes'] = N_CLUSTERS + 1
meta_class['class names'] = class_names_list
meta_class['class lookup'] = [item for sublist in class_colors_list for item in sublist]
meta_class['description'] = 'Structure-Aware Contrastive Mineral Classification'
try:
    if 'wavelength' in meta_class: del meta_class['wavelength']
    if 'fwhm' in meta_class: del meta_class['fwhm']
except: pass

spectral.envi.save_image(CLASS_HDR, full_class_map.reshape(rows, cols, 1), metadata=meta_class, force=True)
print(f"  [+] Saved Classification Map → {CLASS_HDR}")

meta_conf = dict(img.metadata)
meta_conf['bands'] = 1
meta_conf['data type'] = 4 # Float32
meta_conf['description'] = 'Contrastive Embeddings Uncertainty/Confidence Map (%)'
try:
    if 'wavelength' in meta_conf: del meta_conf['wavelength']
except: pass

spectral.envi.save_image(CONF_HDR, full_conf_map.reshape(rows, cols, 1), metadata=meta_conf, force=True)
print(f"  [+] Saved Confidence Map     → {CONF_HDR}")

if n_endmembers > 0:
    meta_abund = dict(img.metadata)
    meta_abund['bands'] = n_endmembers
    meta_abund['data type'] = 4 # Float32
    meta_abund['description'] = 'Subpixel Fractional Abundances (0.0 to 1.0)'
    meta_abund['band names'] = endmember_names
    try:
        if 'wavelength' in meta_abund: del meta_abund['wavelength']
    except: pass
    
    spectral.envi.save_image(ABUND_HDR, abundance_cube, metadata=meta_abund, force=True)
    print(f"  [+] Saved Abundance Map      → {ABUND_HDR}")

# Plotting the matches
fig, axes = plt.subplots(2, (N_CLUSTERS+1)//2, figsize=(18, 10), facecolor='#0d1117')
axes = axes.flatten()

for k in range(N_CLUSTERS):
    ax = axes[k]
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    for sp in ax.spines.values(): sp.set_color('#30363d')
    
    m_spec = cluster_means[k]
    ax.plot(wl, m_spec, label='M3 Cluster Mean', color=cluster_colors[k], lw=2)
    
    # Reload the match to plot it
    match_name = cluster_names[k]
    if match_name in master_lib:
        l_wl, l_ref_cr = master_lib[match_name]
        # Already continuum-removed, no scaling factor needed
        ax.plot(l_wl, l_ref_cr, '--', label=f'RELAB: {match_name}', color='white', alpha=0.7)
        
    ax.set_title(f"Cluster {k+1} Matches {match_name}", color='w', fontsize=10)
    # Truncate plot axis to match where continuum removal is active
    ax.set_xlim(wl.min(), 2500)
    if k==0: ax.legend(facecolor='#0d1117', labelcolor='w')

for ax in axes[N_CLUSTERS:]:
    ax.set_visible(False)

fig.tight_layout()
fig.savefig(PROOF_FIG, dpi=150, facecolor='#0d1117')
plt.close(fig)
print(f"  [+] Saved Verification Plot  → {PROOF_FIG}")

banner("PIPELINE COMPLETE")
