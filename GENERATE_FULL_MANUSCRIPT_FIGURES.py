import os, sys, time, warnings
import numpy as np
import spectral
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
CONF_HDR    = os.path.join(SCENE_DIR, "Classification", f"{SCENE_BASE}_CONFIDENCE.hdr")
ABUND_HDR   = os.path.join(SCENE_DIR, "Classification", f"{SCENE_BASE}_ABUNDANCES.hdr")
METRICS_CSV = os.path.join(SCENE_DIR, "Validation", "bandwise_metrics.csv")

OUT_DIR = os.path.join(SCENE_DIR, "Full_Manuscript_Figures")
os.makedirs(OUT_DIR, exist_ok=True)

try:
    plt.style.use('ggplot')
except Exception:
    pass

def banner(msg): print("\n" + "="*75 + f"\n  {msg}\n" + "="*75, flush=True)

# ── 1. LOAD DATA CUBES ───────────────────────────────────────────────
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

# Find a highly structured pixel
flat_final = cube_final.reshape(-1, bands)
valid_mask = np.isfinite(flat_final[:, 0]) & (flat_final[:, 0] > 0)
valid_indices = np.where(valid_mask)[0]
sample_idx = valid_indices[np.argmax(np.var(flat_final[valid_indices], axis=1))]
r_idx, c_idx = np.unravel_index(sample_idx, (rows, cols))


# ── BLOCK 1: SPATIAL IMAGE EVOLUTION ─────────────────────────────────
banner("Generating Block 1: Spatial & Temporal Maps...")
band_to_plot = 40 # Middle SWIR band where noise is prominent (~1400nm)

# Vis 17: 3-Stage Spatial Evolution
fig17, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
img_r = cube_raw[:, :, band_to_plot]
img_p = cube_phys[:, :, band_to_plot]
img_f = cube_final[:, :, band_to_plot]

vm, vx = np.nanpercentile(img_r, 2), np.nanpercentile(img_r, 98)
axes[0].imshow(img_r, cmap='gray', vmin=vm, vmax=vx)
axes[0].set_title("Visual 17A: RAW (Pre-Processing)")
axes[1].imshow(img_p, cmap='gray', vmin=vm, vmax=vx)
axes[1].set_title("Visual 17B: Physics Corrected")
axes[2].imshow(img_f, cmap='gray', vmin=vm, vmax=vx)
axes[2].set_title("Visual 17C: ML Denoised")
for ax in axes: ax.axis('off')
fig17.tight_layout()
fig17.savefig(os.path.join(OUT_DIR, "Fig17_Spatial_Evolution.png"), dpi=300)
plt.close(fig17)

# Vis 1: Spatial heatmap of noise intensity
fig1, ax1 = plt.subplots(figsize=(8, 8), facecolor='white')
noise_intensity = np.abs(img_r - img_f)
im1 = ax1.imshow(noise_intensity, cmap='inferno')
ax1.set_title("Visual 1: Spatial Heatmap of Pixel-wise Noise Intensity")
ax1.axis('off')
fig1.colorbar(im1, ax=ax1, shrink=0.8, label="Absolute Error (Raw - Denoised)")
fig1.savefig(os.path.join(OUT_DIR, "Fig1_Noise_Intensity.png"), dpi=300)
plt.close(fig1)

# Vis 5 & 8: Localized Spatial Zoom stability
fig5, axes5 = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
crop_r = img_r[rows//2:rows//2+100, cols//2:cols//2+100]
crop_f = img_f[rows//2:rows//2+100, cols//2:cols//2+100]
axes5[0].imshow(crop_r, cmap='gray', vmin=vm, vmax=vx)
axes5[0].set_title("Visual 5: Zoomed RAW Striping")
axes5[1].imshow(crop_f, cmap='gray', vmin=vm, vmax=vx)
axes5[1].set_title("Visual 8: Zoomed FINAL Pixel Stability")
for ax in axes5: ax.axis('off')
fig5.savefig(os.path.join(OUT_DIR, "Fig5_Spatial_Zoom_Stability.png"), dpi=300)
plt.close(fig5)


# ── BLOCK 2: STATISTICAL METRICS (from CSV) ──────────────────────────
banner("Generating Block 2: Statistical Metrics & Stability...")

if os.path.exists(METRICS_CSV):
    df_metrics = pd.read_csv(METRICS_CSV)
    
    # Vis 7 & 9: Spectral Stability curves (SNR and NER)
    fig7, (ax7a, ax7b) = plt.subplots(2, 1, figsize=(12, 10), facecolor='white')
    ax7a.plot(df_metrics['Wavelength_nm'], df_metrics['RAW_SNR'], label='RAW SNR', color='gray', linestyle='--')
    ax7a.plot(df_metrics['Wavelength_nm'], df_metrics['FINAL_SNR'], label='FINAL SNR (ML Denoised)', color='green', linewidth=2)
    ax7a.set_title("Visual 7/9: Signal-to-Noise Ratio (SNR) Stability Across Wavelengths")
    ax7a.set_ylabel("SNR")
    ax7a.legend()
    
    ax7b.plot(df_metrics['Wavelength_nm'], df_metrics['RAW_NER'], label='RAW Spatial Noise (NER %)', color='red', linestyle='--')
    ax7b.plot(df_metrics['Wavelength_nm'], df_metrics['FINAL_NER'], label='FINAL Spatial Noise (NER %)', color='blue', linewidth=2)
    ax7b.axhline(0.5, color='black', linestyle=':', label='Quality Threshold < 0.5%')
    ax7b.set_title("Visual 7/9: Spatial Noise Equivalent Reflectance (NER) across Wavelengths")
    ax7b.set_ylabel("Stripe NER (%)")
    ax7b.set_yscale('log')
    ax7b.set_xlabel("Wavelength (nm)")
    ax7b.legend()
    fig7.tight_layout()
    fig7.savefig(os.path.join(OUT_DIR, "Fig7_9_Stability_Curves.png"), dpi=300)
    plt.close(fig7)
    
    # Vis 2, 3, 6, 11, 12: Bar charts ranking noise
    mean_raw_snr = df_metrics['RAW_SNR'].mean()
    mean_fin_snr = df_metrics['FINAL_SNR'].mean()
    mean_raw_ner = df_metrics['RAW_NER'].mean()
    mean_fin_ner = df_metrics['FINAL_NER'].mean()
    
    fig6, axes6 = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')
    axes6[0].bar(['RAW', 'FINAL'], [mean_raw_snr, mean_fin_snr], color=['gray', 'green'])
    axes6[0].set_title("Visual 6/11: Aggregate SNR Improvement")
    axes6[0].set_ylabel("Mean SNR")
    
    axes6[1].bar(['RAW', 'FINAL'], [mean_raw_ner, mean_fin_ner], color=['red', 'blue'])
    axes6[1].set_title("Visual 2/12: Aggregate Stripe NER Reduction")
    axes6[1].set_ylabel("Mean NER (%)")
    axes6[1].set_yscale('log')
    fig6.tight_layout()
    fig6.savefig(os.path.join(OUT_DIR, "Fig6_11_Global_Metrics_Bars.png"), dpi=300)
    plt.close(fig6)
else:
    print(f"  [!] Missing {METRICS_CSV}. Skipping Block 2.")


# ── BLOCK 3: SPECTRAL CORRECTIONS & ZOOM-INS ─────────────────────────
banner("Generating Block 3: Spectral Comparisons & Zoom-ins...")
spec_raw = cube_raw[r_idx, c_idx, :].copy()
spec_phys = cube_phys[r_idx, c_idx, :].copy()
spec_fin = cube_final[r_idx, c_idx, :].copy()

# Vis 4, 10, 13: Full Spectral Evolutionary Trace
fig4, ax4 = plt.subplots(figsize=(12, 6), facecolor='white')
ax4.plot(wl, spec_raw, label='RAW Spectrum', color='lightgray', linewidth=1)
ax4.plot(wl, spec_phys, label='Physics Corrected (Spike/Thermal Removed)', color='orange', linewidth=1.5, alpha=0.8)
ax4.plot(wl, spec_fin, label='ML Denoised (Final Architecture)', color='black', linewidth=2)
ax4.set_title("Visual 4/10/13: Global Spectral Correction Evolution")
ax4.set_xlabel("Wavelength (nm)")
ax4.set_ylabel("Reflectance")
ax4.legend()
fig4.savefig(os.path.join(OUT_DIR, "Fig4_10_Global_Spectra.png"), dpi=300)
plt.close(fig4)

# Vis 16: Zoomed Spectral Comparison (Publication Quality)
fig16, (ax16a, ax16b) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
ax16a.plot(wl, spec_raw, color='lightgray', marker='o', markersize=3, label='RAW')
ax16a.plot(wl, spec_fin, color='blue', linewidth=2, label='FINAL')
ax16a.set_xlim(800, 1200) # 1 um bowl
ax16a.set_ylim(np.nanmin(spec_fin[(wl>800) & (wl<1200)]) * 0.95, np.nanmax(spec_fin[(wl>800) & (wl<1200)]) * 1.05)
ax16a.set_title("Visual 16A: Diagnostic Accuracy at 1µm Bowl")
ax16a.set_xlabel("Wavelength (nm)")
ax16a.legend()

ax16b.plot(wl, spec_raw, color='lightgray', marker='o', markersize=3, label='RAW (Thermal Tail)')
ax16b.plot(wl, spec_fin, color='red', linewidth=2, label='FINAL (Thermal Corrected)')
ax16b.set_xlim(1800, 2400) # 2 um bowl
ax16b.set_ylim(np.nanmin(spec_fin[(wl>1800) & (wl<2400)]) * 0.95, np.nanmax(spec_fin[(wl>1800) & (wl<2400)]) * 1.05)
ax16b.set_title("Visual 16B: Diagnostic Accuracy & Thermal Suppression at 2µm Bowl")
ax16b.set_xlabel("Wavelength (nm)")
ax16b.legend()
fig16.tight_layout()
fig16.savefig(os.path.join(OUT_DIR, "Fig16_Zoomed_Absorption_Bowls.png"), dpi=300)
plt.close(fig16)


# ── BLOCK 4: MINERALOGY, CONFIDENCE & ISRU ───────────────────────────
banner("Generating Block 4: Composition & ISRU Indicators...")
if os.path.exists(CLASS_HDR) and os.path.exists(ABUND_HDR):
    img_class = spectral.open_image(CLASS_HDR)
    class_map = img_class.read_band(0)
    class_names = img_class.metadata.get('class names', [])
    unique, counts = np.unique(class_map, return_counts=True)
    
    # Vis 14: Pie Chart of Composition
    valid_classes = []
    valid_counts = []
    for u, c in zip(unique, counts):
        if u > 0 and u < len(class_names):
            valid_classes.append(class_names[u])
            valid_counts.append(c)
            
    if valid_counts:
        fig14, ax14 = plt.subplots(figsize=(8, 8), facecolor='white')
        ax14.pie(valid_counts, labels=valid_classes, autopct='%1.1f%%', startangle=140, colors=plt.get_cmap("Set2").colors)
        ax14.set_title("Visual 14: Bulk Crater Mineral Composition")
        fig14.savefig(os.path.join(OUT_DIR, "Fig14_Mineral_Pie.png"), dpi=300)
        plt.close(fig14)

    # Vis 15: Conidence Heatmap
    if os.path.exists(CONF_HDR):
        img_conf = spectral.open_image(CONF_HDR)
        conf_map = img_conf.read_band(0)
        fig15, ax15 = plt.subplots(figsize=(10, 8), facecolor='white')
        im15 = ax15.imshow(conf_map, cmap='inferno', vmin=0, vmax=100)
        ax15.set_title("Visual 15: Latent-Space Matching Confidence Heatmap")
        ax15.axis('off')
        fig15.colorbar(im15, ax=ax15, label="Cosine Similarity Confidence (%)")
        fig15.savefig(os.path.join(OUT_DIR, "Fig15_Confidence_Map.png"), dpi=300)
        plt.close(fig15)

    # Vis 18: ISRU Indicators
    img_abund = spectral.open_image(ABUND_HDR)
    abund_cube = img_abund.load()
    abund_names = img_abund.metadata.get('band names', [f"Mineral {i}" for i in range(abund_cube.shape[2])])
    
    # Separate Al/Ca (Plagioclase/Albite) vs Fe/Mg (Pyroxene/Olivine/Diopside)
    plag_idx = [i for i, n in enumerate(abund_names) if 'plagioclase' in n.lower() or 'albite' in n.lower() or 'anorthosite' in n.lower()]
    mafic_idx = [i for i, n in enumerate(abund_names) if 'pyroxene' in n.lower() or 'olivine' in n.lower() or 'diopside' in n.lower() or 'enstatite' in n.lower()]
    
    if plag_idx or mafic_idx:
        fig18, axes18 = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
        
        if plag_idx:
            plag_map = np.sum(abund_cube[:,:,plag_idx], axis=2)
            im_p = axes18[0].imshow(plag_map, cmap='Blues', vmin=0, vmax=1)
            axes18[0].set_title("Visual 18A: Aluminum/Silicon ISRU Indicator (Plagioclase sum)")
            fig18.colorbar(im_p, ax=axes18[0])
        axes18[0].axis('off')
            
        if mafic_idx:
            mafic_map = np.sum(abund_cube[:,:,mafic_idx], axis=2)
            im_m = axes18[1].imshow(mafic_map, cmap='OrRd', vmin=0, vmax=1)
            axes18[1].set_title("Visual 18B: Iron/Magnesium ISRU Indicator (Pyroxene/Olivine sum)")
            fig18.colorbar(im_m, ax=axes18[1])
        axes18[1].axis('off')

        fig18.tight_layout()
        fig18.savefig(os.path.join(OUT_DIR, "Fig18_ISRU_Indicators.png"), dpi=300)
        plt.close(fig18)

banner("FULL MANUSCRIPT FIGURES COMPLETED")
print(f"Check the directory: {OUT_DIR}")
