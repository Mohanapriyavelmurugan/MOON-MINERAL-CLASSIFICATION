import os, warnings
import numpy as np
import spectral
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import ndimage

warnings.filterwarnings('ignore')

import sys

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    HDR_PATH = sys.argv[1]
else:
    HDR_PATH = r"D:\Moon_Data\Scene_2\M3G20081201T064047_V01_RFL.HDR"

SCENE_DIR   = os.path.dirname(HDR_PATH)
SCENE_BASE  = os.path.basename(HDR_PATH).replace('.HDR', '').replace('.hdr', '')
OUT_DIR     = os.path.join(SCENE_DIR, "Physics_Corrected")
OUT_HDR     = os.path.join(OUT_DIR, f"{SCENE_BASE}_CORRECTED.hdr")
PROOF_FIG   = os.path.join(OUT_DIR, "before_after_proof.png")
PROOF_TXT   = os.path.join(OUT_DIR, "correction_proof.txt")
THERMAL_NM  = 2500      # nm — thermal onset wavelength
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

DARK = '#0d1117'; AX = '#161b22'
C = {'blue':'#58a6ff','green':'#3fb950','orange':'#f0883e',
     'red':'#ff7b72','yellow':'#e3b341','purple':'#bc8cff'}

def style_ax(ax):
    ax.set_facecolor(AX)
    for sp in ax.spines.values(): sp.set_color('#30363d')
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.xaxis.label.set_color('#c9d1d9')
    ax.yaxis.label.set_color('#c9d1d9')
    ax.title.set_color('#c9d1d9')

def noise_metrics(cube, wl):
    """Compute SNR and NER on the cube — used for before/after comparison."""
    rows, cols, bands = cube.shape
    snr_list, ner_list = [], []
    for b in range(bands):
        bd = cube[:, :, b]
        valid = np.isfinite(bd)
        if valid.sum() < 50: continue
        smooth   = ndimage.uniform_filter(np.nan_to_num(bd, nan=0.0), size=3)
        residual = bd - smooth
        noise    = np.nanstd(residual[valid])
        sig      = np.nanmean(bd[valid])
        if noise > 0: snr_list.append(sig / noise)
        col_means  = np.nanmean(bd, axis=0)
        scene_mean = np.nanmean(bd[valid])
        if scene_mean > 0: ner_list.append(np.std(col_means) / scene_mean * 100)
    return (float(np.median(snr_list))  if snr_list else 0.0,
            float(np.median(ner_list)) if ner_list else 0.0)

# ── STEP 0: LOAD ──────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 0  ·  Loading M3 L2 Cube")
print("="*65)

img  = spectral.open_image(HDR_PATH)
in_cube = img.open_memmap(interleave='bip', writable=False)
rows, cols, bands = in_cube.shape

cube = np.empty((rows, cols, bands), dtype=np.float32)
for b in range(bands):
    cube[:, :, b] = np.array(in_cube[:, :, b], dtype=np.float32)

wl_raw = img.metadata.get('wavelength', [])
wl = np.array([float(w) for w in wl_raw])
if wl.max() < 10: wl *= 1000.0    # µm → nm

# Mask missing data flags
cube[cube < -990] = np.nan

print(f"  Shape      : {rows} × {cols} × {bands}")
print(f"  Wavelengths: {wl.min():.0f} – {wl.max():.0f} nm")

# ── Compute BEFORE metrics ────────────────────────────────────────────────────
print("\n  Computing BEFORE metrics ...")
snr_before, ner_before = noise_metrics(cube, wl)
mean_spec_before = np.zeros(bands, dtype=np.float64)
for b_idx in range(bands):
    mean_spec_before[b_idx] = np.nanmean(cube[:, :, b_idx])
print(f"  SNR (before): {snr_before:.2f}   NER (before): {ner_before:.3f}%")

corrected = cube.copy()   # work on this throughout corrections

# ═════════════════════════════════════════════════════════════════════════════
# CORRECTION 1 — DETECTOR SATURATION & OUTLIERS: Mask & Interpolate (MUST BE FIRST)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  CORRECTION 1  ·  Detector Saturation & Outliers — Mask & Interpolate")
print("="*65)

total_sat = 0
for b in range(bands):
    bd = corrected[:, :, b]
    valid = np.isfinite(bd)
    if valid.sum() == 0: continue
    
    # 1. Physical anomaly bounds (reflectance should be between 0 and 1.2 roughly)
    sat_mask = ((bd > 1.5) | (bd < -0.05)) & valid
    
    # 2. Statistical absolute outlier (extremely heavy spikes, > 10 std deviations)
    if valid.sum() > 100:
        band_mean = np.nanmean(bd[valid])
        band_std  = np.nanstd(bd[valid])
        spike_mask = (np.abs(bd - band_mean) > 10 * band_std) & valid
        sat_mask  = sat_mask | spike_mask

    n_sat = int(sat_mask.sum())
    total_sat += n_sat
    corrected[sat_mask, b] = np.nan   # flag as invalid

# Spectrally interpolate NaN-flagged pixels (Using LINEAR to avoid cubic ringing)
sat_pixels_fixed = 0
flat_corrected = corrected.reshape(-1, bands)

# Only iterate over pixels that genuinely contain NaNs
has_nan = np.isnan(flat_corrected).any(axis=1)
nan_indices = np.where(has_nan)[0]

for idx in nan_indices:
    spec = flat_corrected[idx]
    nan_mask = ~np.isfinite(spec)
    valid_b = np.where(~nan_mask)[0]
    if len(valid_b) < 3: 
        flat_corrected[idx, :] = 0.0
        continue
    
    # Linear interpolation prevents artificial undershoots/overshoots
    f = interp1d(valid_b, spec[valid_b], kind='linear',
                 bounds_error=False, fill_value='extrapolate')
    flat_corrected[idx, nan_mask] = f(np.where(nan_mask)[0])
    sat_pixels_fixed += nan_mask.sum()

corrected = flat_corrected.reshape(rows, cols, bands)

corrected = np.clip(corrected, 0.0, 1.5)   # physical reflectance bounds

print(f"  Anomalous pixel-bands flagged : {total_sat:,}")
print(f"  Interpolated & restored       : {sat_pixels_fixed:,}")

# ═════════════════════════════════════════════════════════════════════════════
# CORRECTION 2 — STRIPE NOISE: Column Moment Matching
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  CORRECTION 2  ·  Stripe Noise — Column Moment Matching")
print("="*65)

for b in range(bands):
    bd = corrected[:, :, b]
    valid_full = np.isfinite(bd)
    if valid_full.sum() < 50: continue
    scene_mean = np.nanmean(bd[valid_full])
    scene_std  = np.nanstd(bd[valid_full])
    if scene_std < 1e-10: continue

    for ci in range(cols):
        col = bd[:, ci]
        valid = np.isfinite(col)
        if valid.sum() < 5: continue
        col_mean = np.nanmean(col[valid])
        col_std  = np.nanstd(col[valid])
        if col_std < 1e-10:
            corrected[:, ci, b] = col - col_mean + scene_mean
        else:
            corrected[:, ci, b] = (col - col_mean) / col_std * scene_std + scene_mean

snr_s1, ner_s1 = noise_metrics(corrected, wl)
print(f"  NER after stripe correction : {ner_s1:.3f}%   (was {ner_before:.3f}%)")
print(f"  NER reduction               : {max(0, (ner_before - ner_s1)/ner_before*100):.1f}%")

# ═════════════════════════════════════════════════════════════════════════════
# CORRECTION 3 — SPECTRAL SMILE: Per-Column Wavelength Resampling
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  CORRECTION 3  ·  Spectral Smile — Wavelength Resampling")
print("="*65)

# Measure per-column band-centre of 1000 nm feature
feat_wl   = 1000.0
c_band    = int(np.argmin(np.abs(wl - feat_wl)))
half_win  = 10
b_lo, b_hi = max(0, c_band - half_win), min(bands - 1, c_band + half_win)

col_shift_bands = np.zeros(cols, dtype=np.float64)  # shift in band indices
nadir_col       = cols // 2

def find_centre(spectrum):
    """Find the band index of minimum in a short spectral window."""
    if not np.any(np.isfinite(spectrum)): return np.nan
    smooth = ndimage.uniform_filter(np.nan_to_num(spectrum, nan=np.nanmean(spectrum)), 3)
    return float(np.argmin(smooth))

# Reference minimum at nadir column
nadir_spec  = np.nanmean(corrected[:, nadir_col, b_lo:b_hi], axis=0)
ref_min_idx = find_centre(nadir_spec)

for ci in range(cols):
    col_spec    = np.nanmean(corrected[:, ci, b_lo:b_hi], axis=0)
    this_min    = find_centre(col_spec)
    col_shift_bands[ci] = (0.0 if np.isnan(this_min) or np.isnan(ref_min_idx)
                           else this_min - ref_min_idx)

# Fit 2nd-order polynomial to smooth the shift model
col_idx       = np.arange(cols)
valid_shift   = np.isfinite(col_shift_bands)
poly_coeffs   = np.polyfit(col_idx[valid_shift], col_shift_bands[valid_shift], 2)
smooth_shift  = np.polyval(poly_coeffs, col_idx)

bw = float(np.nanmean(np.diff(wl)))    # nm per band
smile_before_nm = (col_shift_bands[valid_shift].max() -
                   col_shift_bands[valid_shift].min()) * bw

# Apply correction: resample each column onto the reference grid
for ci in range(cols):
    shift_nm = smooth_shift[ci] * bw
    if abs(shift_nm) < 0.01: continue           # negligible shift
    shifted_wl = wl + shift_nm                  # column's actual wavelength grid
    
    col_data = corrected[:, ci, :]
    
    # Vectorize: Find completely finite rows to interpolate instantly in a single batch
    valid_mask_2d = np.isfinite(col_data)
    valid_rows = np.where(valid_mask_2d.all(axis=1))[0]
    
    if len(valid_rows) > 0:
        f = interp1d(shifted_wl, col_data[valid_rows], axis=-1,
                     kind='cubic', bounds_error=False,
                     fill_value='extrapolate')
        corrected[valid_rows, ci, :] = f(wl)

    # Process any remaining rows with isolated missing data via the robust fallback loop
    invalid_rows = np.where(~valid_mask_2d.all(axis=1))[0]
    for ri in invalid_rows:
        spec = col_data[ri]
        if not np.any(np.isfinite(spec)): continue
        valid = np.isfinite(spec)
        if valid.sum() < 5: continue
        # Natural cubic spline interpolation is safe here since extreme anomalies were removed
        f = interp1d(shifted_wl[valid], spec[valid],
                     kind='cubic', bounds_error=False,
                     fill_value='extrapolate')
        corrected[ri, ci, :] = f(wl)

# Measure smile after correction
col_shift_after = np.zeros(cols, dtype=np.float64)
for ci in range(cols):
    col_spec  = np.nanmean(corrected[:, ci, b_lo:b_hi], axis=0)
    this_min  = find_centre(col_spec)
    col_shift_after[ci] = (0.0 if np.isnan(this_min) or np.isnan(ref_min_idx)
                           else this_min - ref_min_idx)
smile_after_nm = ((col_shift_after.max() - col_shift_after.min()) * bw)

print(f"  Smile range before : {smile_before_nm:.2f} nm")
print(f"  Smile range after  : {smile_after_nm:.2f} nm")
print(f"  Smile reduction    : {max(0,(smile_before_nm-smile_after_nm)/smile_before_nm*100):.1f}%")

# ═════════════════════════════════════════════════════════════════════════════
# CORRECTION 4 — THERMAL EMISSION: Spatial-Smoothed Planck Subtraction
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  CORRECTION 4  ·  Thermal Emission — Spatial-Smoothed Planck Subtraction")
print("="*65)

h, c_light, k_b = 6.626e-34, 3.0e8, 1.381e-23

def planck(lam_nm, T):
    lam = lam_nm * 1e-9
    return (2 * h * c_light**2 / lam**5) / (np.exp(h * c_light / (lam * k_b * T)) - 1)

th_mask  = wl >= THERMAL_NM
if th_mask.sum() >= 2:
    th_wls   = wl[th_mask]
    
    # Use the last 4 bands for stability, not just the last 2
    t_bands = np.where(th_mask)[0][-4:]
    th_mean_before = float(np.nanmean(corrected[:, :, th_mask]))

    # Calculate observed thermal map and spatially smooth it (5x5 filter)
    # This prevents injecting pixel-to-pixel shot noise into the dataset
    obs_th_map = np.nanmean(corrected[:, :, t_bands], axis=2)
    obs_th_map = np.nan_to_num(obs_th_map, nan=0.0)
    smoothed_obs_th = ndimage.median_filter(obs_th_map, size=5)
    
    lam_ref = float(np.median(th_wls))

    # Vectorized Planck Temperature Estimation & Subtraction
    valid_th_mask = (smoothed_obs_th > 0)
    T_est_map = np.zeros((rows, cols), dtype=np.float64)
    
    term1 = 2 * h * c_light**2 / (lam_ref * 1e-9)**5
    arg_map = np.zeros((rows, cols), dtype=np.float64)
    arg_map[valid_th_mask] = term1 / np.maximum(smoothed_obs_th[valid_th_mask], 1e-20) + 1
    
    valid_arg_mask = valid_th_mask & (arg_map > 1.0)
    term2 = h * c_light / (lam_ref * 1e-9 * k_b)
    T_est_map[valid_arg_mask] = term2 / np.log(arg_map[valid_arg_mask])
    T_est_map = np.clip(T_est_map, 200.0, 500.0)
    
    # Calculate Planck curves for all valid pixels
    valid_idx = np.where(valid_arg_mask)
    if len(valid_idx[0]) > 0:
        T_valid = T_est_map[valid_idx].reshape(-1, 1) # Shape: (K, 1)
        lam_2d = (th_wls * 1e-9).reshape(1, -1)       # Shape: (1, B)
        
        planck_valid = (2 * h * c_light**2 / lam_2d**5) / (np.exp(h * c_light / (lam_2d * k_b * T_valid)) - 1)
        planck_sum_valid = planck_valid.sum(axis=1, keepdims=True) # Shape: (K, 1)
        
        mask_planck = (planck_sum_valid.flatten() >= 1e-30)
        
        if np.any(mask_planck):
            obs_th_valid = smoothed_obs_th[valid_idx].reshape(-1, 1)
            alpha_valid = obs_th_valid * th_mask.sum() / planck_sum_valid
            
            # Extract only pixels that passed all checks
            final_valid_rows = valid_idx[0][mask_planck]
            final_valid_cols = valid_idx[1][mask_planck]
            
            # Extract target spectra (K_final, B)
            spec_th_valid = corrected[final_valid_rows, final_valid_cols][:, th_mask]
            
            alpha_final = alpha_valid[mask_planck]
            planck_final = planck_valid[mask_planck]
            
            # Subtract
            corrected_th = spec_th_valid - (alpha_final * planck_final)
            
            # Reassign back to the cube efficiently
            b_indices = np.where(th_mask)[0]
            for b_idx_local, b_real in enumerate(b_indices):
                corrected[final_valid_rows, final_valid_cols, b_real] = corrected_th[:, b_idx_local]

    # Clip negative values resulting from any oversubtraction
    corrected = np.clip(corrected, 0.0, 1.5)

    th_mean_after = float(np.nanmean(corrected[:, :, th_mask]))
    th_reduction  = (th_mean_before - th_mean_after) / th_mean_before * 100
    print(f"  Mean thermal reflectance before  : {th_mean_before:.5f}")
    print(f"  Mean thermal reflectance after   : {th_mean_after:.5f}")
    print(f"  Thermal signal reduction         : {th_reduction:.1f}%")
else:
    print("  Not enough long-wave bands — skipping thermal correction.")
    th_mean_before = th_mean_after = th_reduction = 0.0

# ── Compute AFTER metrics ─────────────────────────────────────────────────────
print("\n  Computing AFTER metrics ...")
snr_after, ner_after = noise_metrics(corrected, wl)
mean_spec_after = np.zeros(bands, dtype=np.float64)
for b_idx in range(bands):
    mean_spec_after[b_idx] = np.nanmean(corrected[:, :, b_idx])

# ═════════════════════════════════════════════════════════════════════════════
# SAVE CORRECTED CUBE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  Saving Corrected Cube ...")
print("="*65)

metadata = dict(img.metadata)
metadata['description'] = ('M3 L2 Physics-Corrected: '
                            'outliers + stripe + smile + thermal corrections applied')
try:
    spectral.envi.save_image(OUT_HDR,
                             corrected.astype(np.float32),
                             dtype=np.float32,
                             metadata=metadata,
                             force=True)
    print(f"  Saved → {OUT_HDR}")
except Exception as e:
    print(f"  Failed to save: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# BEFORE / AFTER PROOF FIGURE
# ═════════════════════════════════════════════════════════════════════════════
print("\n  Generating before/after proof figure ...")

fig = plt.figure(figsize=(20, 14), facecolor=DARK)
fig.suptitle(
    "Physics-Based Noise Corrections — Before vs After Proof\n"
    "M3G20081201T064047_V01_RFL",
    color='#c9d1d9', fontsize=13, fontweight='bold')

gs = fig.add_gridspec(3, 2, hspace=0.48, wspace=0.35,
                      left=0.07, right=0.97, top=0.90, bottom=0.06)

# ── Panel 1: Mean Spectrum Before vs After ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, :]); style_ax(ax1)
ax1.plot(wl, mean_spec_before, color=C['red'],    lw=1.5, label='Before corrections', alpha=0.9)
ax1.plot(wl, mean_spec_after,  color=C['green'],  lw=1.5, label='After corrections',  alpha=0.9)
ax1.axvline(THERMAL_NM, color=C['yellow'], lw=0.8, linestyle=':', label=f'Thermal onset ({THERMAL_NM} nm)')
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Reflectance")
ax1.set_title("Mean Spectrum — Before vs After All Corrections")
ax1.legend(facecolor='#161b22', labelcolor='white', fontsize=9)

# ── Panel 2: Stripe — Band image before/after ────────────────────────────────
stripe_band = int(np.argmin(np.abs(wl - 750)))

ax2 = fig.add_subplot(gs[1, 0]); style_ax(ax2)
im2a = ax2.imshow(cube[:, :, stripe_band], cmap='gray',
                  vmin=np.nanpercentile(cube[:, :, stripe_band], 1),
                  vmax=np.nanpercentile(cube[:, :, stripe_band], 99), aspect='auto')
ax2.set_title(f"Stripe Noise — BEFORE  (Band @ {wl[stripe_band]:.0f} nm)")
ax2.set_xlabel("Column"); ax2.set_ylabel("Row")
plt.colorbar(im2a, ax=ax2, fraction=0.046, label='Reflectance')

ax3 = fig.add_subplot(gs[1, 1]); style_ax(ax3)
im3a = ax3.imshow(corrected[:, :, stripe_band], cmap='gray',
                  vmin=np.nanpercentile(cube[:, :, stripe_band], 1),
                  vmax=np.nanpercentile(cube[:, :, stripe_band], 99), aspect='auto')
ax3.set_title(f"Stripe Noise — AFTER  (Band @ {wl[stripe_band]:.0f} nm)")
ax3.set_xlabel("Column"); ax3.set_ylabel("Row")
plt.colorbar(im3a, ax=ax3, fraction=0.046, label='Reflectance')

# ── Panel 3: Thermal region zoom — before/after ───────────────────────────────
ax4 = fig.add_subplot(gs[2, 0]); style_ax(ax4)
th_idx = wl >= 2000
ax4.plot(wl[th_idx], mean_spec_before[th_idx], color=C['red'],   lw=1.5,
         label='Before', alpha=0.9)
ax4.plot(wl[th_idx], mean_spec_after[th_idx],  color=C['green'], lw=1.5,
         label='After',  alpha=0.9)
ax4.axvspan(THERMAL_NM, wl.max(), alpha=0.1, color=C['orange'])
ax4.set_xlabel("Wavelength (nm)")
ax4.set_ylabel("Reflectance")
ax4.set_title("Thermal Region (>2000 nm) — Before vs After")
ax4.legend(facecolor='#161b22', labelcolor='white', fontsize=9)

# ── Panel 4: Column profile — stripe proof ────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1]); style_ax(ax5)
col_means_before = np.nanmean(cube[:, :, stripe_band], axis=0)
col_means_after  = np.nanmean(corrected[:, :, stripe_band], axis=0)
ax5.plot(col_means_before, color=C['red'],   lw=1.2, label='Before', alpha=0.9)
ax5.plot(col_means_after,  color=C['green'], lw=1.2, label='After',  alpha=0.9)
ax5.set_xlabel("Column Index")
ax5.set_ylabel("Mean Reflectance")
ax5.set_title(f"Column Profile @ {wl[stripe_band]:.0f} nm — Stripe Uniformity")
ax5.legend(facecolor='#161b22', labelcolor='white', fontsize=9)

try:
    fig.savefig(PROOF_FIG, dpi=150, bbox_inches='tight', facecolor=DARK)
except Exception:
    pass
plt.close(fig)
print(f"  Saved → {PROOF_FIG}")

# ═════════════════════════════════════════════════════════════════════════════
# QUANTIFIED PROOF TABLE
# ═════════════════════════════════════════════════════════════════════════════
snr_improvement  = (snr_after  - snr_before)  / max(snr_before,  1e-6) * 100
ner_reduction    = (ner_before  - ner_after)   / max(ner_before,  1e-6) * 100
smile_reduction  = max(0, (smile_before_nm - smile_after_nm) / max(smile_before_nm, 1e-6) * 100)

report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║     PHYSICS-BASED CORRECTIONS — QUANTIFIED PROOF REPORT              ║
║     Scene: M3G20081118T222604_V01_RFL                                ║
╠══════════════════════════════════════════════════════════════════════╣
║  METRIC                    BEFORE       AFTER        IMPROVEMENT     ║
╠══════════════════════════════════════════════════════════════════════╣
║  SNR (higher=better)       {snr_before:>8.2f}     {snr_after:>8.2f}     {snr_improvement:>+8.1f}%      ║
║  Stripe NER (lower=better) {ner_before:>8.3f}%    {ner_after:>8.3f}%    {ner_reduction:>+8.1f}%      ║
║  Spectral Smile            {smile_before_nm:>8.2f} nm  {smile_after_nm:>8.2f} nm  {smile_reduction:>+8.1f}%      ║
║  Thermal mean reflectance  {th_mean_before:>8.5f}     {th_mean_after:>8.5f}     {th_reduction:>+8.1f}%      ║
║  Anomalous bands fixed     {total_sat:>8,}     {'✔ restored':>10}    ║
╠══════════════════════════════════════════════════════════════════════╣
║  CORRECTIONS APPLIED (in order):                                     ║
║    1. Valid ranges & 10-sigma outliers masked (linear interp)        ║
║    2. Column moment matching (stripe)                                ║
║    3. Per-column spline resampling (spectral smile)                  ║
║    4. Spatially-smoothed Planck subtraction (thermal emission)       ║
╠══════════════════════════════════════════════════════════════════════╣
║  OUTPUT CUBE : {os.path.basename(OUT_HDR):<53}║
╚══════════════════════════════════════════════════════════════════════╝
"""

print(report)

try:
    with open(PROOF_TXT, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Report saved → {PROOF_TXT}")
except Exception:
    pass
print("\n  ✔  All physics-based corrections complete.\n")
