"""
M3 L2 Noise Characterisation — Simple Classification Report
============================================================
For each noise type, computes one key metric and classifies:
  NONE / LOW / MEDIUM / HIGH
Outputs: console report + noise_report.txt
"""

import numpy as np
import spectral
import warnings
from scipy import ndimage, stats

warnings.filterwarnings('ignore')

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
HDR_PATH    = r"D:\Moon_Data\Scene_2\M3G20081201T064047_V01_RFL.HDR"
REPORT_PATH = r"D:\Moon_Data\Scene_2\Noise_Characterisation\noise_report.txt"
THERMAL_NM  = 2500   # thermal emission onset wavelength (nm)
SPIKE_THR   = 3.5    # MAD Z-score threshold for spike detection
# ──────────────────────────────────────────────────────────────────────────────

def classify(value, low, medium, high):
    """Return NONE/LOW/MEDIUM/HIGH given thresholds."""
    if   value < low:    return "NONE"
    elif value < medium: return "LOW"
    elif value < high:   return "MEDIUM"
    else:                return "HIGH"

def bar(level):
    bars = {"NONE": "░░░░", "LOW": "▓░░░", "MEDIUM": "▓▓░░", "HIGH": "▓▓▓▓"}
    return bars.get(level, "????")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("\nLoading M3 L2 cube ...")
img  = spectral.open_image(HDR_PATH)
cube = np.array(img.load(), dtype=np.float32)
rows, cols, bands = cube.shape

# Wavelengths
wl_raw = img.metadata.get('wavelength', [])
wavelengths = np.array([float(w) for w in wl_raw])
if wavelengths.max() < 10:
    wavelengths *= 1000.0       # µm → nm

# Replace fill values
cube[cube < -990] = np.nan

print(f"  Shape      : {rows} × {cols} × {bands}")
print(f"  Wavelengths: {wavelengths.min():.0f} – {wavelengths.max():.0f} nm\n")

results = {}   # noise_name -> (metric_value, level, description)

# ══════════════════════════════════════════════════════════════════════════════
# 1. GAUSSIAN / SHOT NOISE  →  Signal-to-Noise Ratio (SNR)
# ══════════════════════════════════════════════════════════════════════════════
snr_list = []
for b in range(bands):
    bd = cube[:, :, b]
    valid = np.isfinite(bd)
    if valid.sum() < 50:
        continue
    smooth   = ndimage.uniform_filter(np.nan_to_num(bd, nan=0.0), size=3)
    residual = bd - smooth
    noise    = np.nanstd(residual[valid])
    sig      = np.nanmean(bd[valid])
    if noise > 0:
        snr_list.append(sig / noise)

median_snr = float(np.median(snr_list)) if snr_list else 0.0
# SNR thresholds: >100=NONE, 50-100=LOW, 20-50=MEDIUM, <20=HIGH noise
g_level = classify(median_snr, 20, 50, 100)   # inverted: lower SNR = higher noise
# Re-map: HIGH SNR = NONE noise
snr_noise_map = {"NONE": "HIGH", "LOW": "MEDIUM", "MEDIUM": "LOW", "HIGH": "NONE"}
g_level = snr_noise_map[g_level]
results["1. Gaussian / Shot Noise"] = (
    median_snr, g_level,
    f"Median SNR = {median_snr:.1f}  |  Typical good range: 100–400"
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. STRIPE NOISE  →  Noise Equivalent Reflectance (NER)
# ══════════════════════════════════════════════════════════════════════════════
ner_list = []
for b in range(bands):
    bd = cube[:, :, b]
    valid = np.isfinite(bd)
    if valid.sum() < 50:
        continue
    col_means  = np.nanmean(bd, axis=0)
    scene_mean = np.nanmean(bd[valid])
    if scene_mean > 0:
        ner_list.append(np.std(col_means) / scene_mean * 100)   # in %

median_ner = float(np.median(ner_list)) if ner_list else 0.0
# NER thresholds: <0.5%=NONE, 0.5-1%=LOW, 1-2%=MEDIUM, >2%=HIGH
s_level = classify(median_ner, 0.5, 1.0, 2.0)
results["2. Stripe Noise"] = (
    median_ner, s_level,
    f"Median NER = {median_ner:.3f}%  |  Acceptable: <0.5%"
)

# ══════════════════════════════════════════════════════════════════════════════
# 3. SPECTRAL SPIKES  →  Spike-affected pixel fraction (%)
# ══════════════════════════════════════════════════════════════════════════════
step      = max(1, rows // 80)        # sample ~80 rows
total_px  = 0
spike_px  = 0

for ri in range(0, rows, step):
    for ci in range(0, cols, 4):      # sample every 4th column
        spec = cube[ri, ci, :].astype(np.float64)
        if not np.any(np.isfinite(spec)):
            continue
        total_px += 1
        med = np.nanmedian(spec)
        mad = np.nanmedian(np.abs(spec - med))
        if mad < 1e-10:
            continue
        mz = np.abs(0.6745 * (spec - med) / mad)
        if np.any(mz > SPIKE_THR):
            spike_px += 1

spike_pct = (spike_px / total_px * 100) if total_px > 0 else 0.0
# Spike thresholds: <1%=NONE, 1-5%=LOW, 5-15%=MEDIUM, >15%=HIGH
sp_level = classify(spike_pct, 1.0, 5.0, 15.0)
results["3. Spectral Spikes (Impulse)"] = (
    spike_pct, sp_level,
    f"Spike-affected pixels = {spike_pct:.2f}%  |  Acceptable: <1%"
)

# ══════════════════════════════════════════════════════════════════════════════
# 4. THERMAL EMISSION RESIDUAL  →  Long-wave reflectance elevation (%)
# ══════════════════════════════════════════════════════════════════════════════
th_mask   = wavelengths >= THERMAL_NM
vnir_mask = (wavelengths >= 500) & (wavelengths <= 1200)

mean_spec = np.nanmean(cube.reshape(-1, bands), axis=0)

if th_mask.sum() >= 2 and vnir_mask.sum() >= 3:
    vnir_mean  = np.nanmean(mean_spec[vnir_mask])
    th_mean    = np.nanmean(mean_spec[th_mask])
    th_excess  = max(0.0, (th_mean - vnir_mean * 0.3) / vnir_mean * 100)
    # Thresholds: <5%=NONE, 5-15%=LOW, 15-30%=MEDIUM, >30%=HIGH
    t_level = classify(th_excess, 5.0, 15.0, 30.0)
    results["4. Thermal Emission Residual"] = (
        th_excess, t_level,
        f"Long-wave elevation above continuum = {th_excess:.1f}%  |  Acceptable: <5%"
    )
else:
    results["4. Thermal Emission Residual"] = (
        0.0, "NONE", "Not enough long-wave bands to assess."
    )

# ══════════════════════════════════════════════════════════════════════════════
# 5. SPECTRAL SMILE  →  Band-centre wavelength shift across swath (nm)
# ══════════════════════════════════════════════════════════════════════════════
feat_wl   = 1000    # pyroxene ~1 µm — most robust feature in M3
c_band    = int(np.argmin(np.abs(wavelengths - feat_wl)))
half_win  = 8
b_lo, b_hi = max(0, c_band - half_win), min(bands - 1, c_band + half_win)
bw        = float(np.nanmean(np.diff(wavelengths)))

col_centres = []
for ci in range(cols):
    spec_col = np.nanmean(cube[:, ci, b_lo:b_hi], axis=0)
    if not np.any(np.isfinite(spec_col)):
        continue
    deriv = np.gradient(spec_col)
    zcs   = np.where(np.diff(np.sign(deriv)) > 0)[0]
    if len(zcs) > 0:
        col_centres.append(zcs[0])

smile_range_nm = 0.0
if len(col_centres) > 5:
    smile_range_nm = (max(col_centres) - min(col_centres)) * bw

# Thresholds: <1 nm=NONE, 1-3=LOW, 3-6=MEDIUM, >6=HIGH
sm_level = classify(smile_range_nm, 1.0, 3.0, 6.0)
results["5. Spectral Smile"] = (
    smile_range_nm, sm_level,
    f"Absorption centre shift = {smile_range_nm:.2f} nm  |  Acceptable: <1 nm"
)

# ══════════════════════════════════════════════════════════════════════════════
# 6. DETECTOR SATURATION  →  Max per-band saturation fraction (%)
# ══════════════════════════════════════════════════════════════════════════════
max_sat_frac = 0.0
for b in range(bands):
    bd    = cube[:, :, b]
    valid = np.isfinite(bd)
    if valid.sum() == 0:
        continue
    thresh = 0.98 * np.nanpercentile(bd[valid], 99.9)
    frac   = (bd[valid] > thresh).sum() / valid.sum() * 100
    if frac > max_sat_frac:
        max_sat_frac = frac

# Thresholds: <0.1%=NONE, 0.1-1%=LOW, 1-5%=MEDIUM, >5%=HIGH
sat_level = classify(max_sat_frac, 0.1, 1.0, 5.0)
results["6. Detector Saturation"] = (
    max_sat_frac, sat_level,
    f"Max band saturation fraction = {max_sat_frac:.3f}%  |  Acceptable: <0.1%"
)

# ══════════════════════════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {
    "NONE":   "\033[92m",   # green
    "LOW":    "\033[93m",   # yellow
    "MEDIUM": "\033[33m",   # orange
    "HIGH":   "\033[91m",   # red
}
RESET = "\033[0m"

report_lines = []
header = (
    "\n"
    "╔══════════════════════════════════════════════════════════════════════╗\n"
    "║     M3 L2 NOISE CHARACTERISATION REPORT                              ║\n"
    "║     Scene: M3G20081201T064047_V01_RFL                                ║\n"
    "╠══════════════════════════════════════════════════════════════════════╣\n"
   f"║  {'NOISE TYPE':<35} {'LEVEL':<8} {'SEVERITY BAR':<10}║\n"
    "╠══════════════════════════════════════════════════════════════════════╣"
)
print(header)
report_lines.append(header.replace("\033[92m","").replace("\033[93m","")
                    .replace("\033[33m","").replace("\033[91m","").replace("\033[0m",""))

for name, (val, level, desc) in results.items():
    col = COLORS.get(level, "")
    row = f"║  {name:<35} {col}{level:<8}{RESET} {bar(level):<10}║"
    print(row)
    report_lines.append(f"║  {name:<35} {level:<8} {bar(level):<10}║")

sep = "╠══════════════════════════════════════════════════════════════════════╣"
print(sep)
report_lines.append(sep)

print(f"║  {'DETAIL':<68}║")
report_lines.append(f"║  {'DETAIL':<68}║")

for name, (val, level, desc) in results.items():
    col   = COLORS.get(level, "")
    short = f"  {name}: {col}{level}{RESET} — {desc}"
    print(f"  {name}: {col}{level}{RESET} — {desc}")
    report_lines.append(f"  {name}: {level} — {desc}")

footer = "╚══════════════════════════════════════════════════════════════════════╝\n"
print(footer)
report_lines.append(footer)

# ── MINERAL IMPACT SUMMARY ────────────────────────────────────────────────────
impact = [
    "\nMINERAL IDENTIFICATION IMPACT SUMMARY:",
    "─" * 50,
]
for name, (val, level, desc) in results.items():
    if level == "NONE":
        verdict = "✔  No impact on mineral identification."
    elif level == "LOW":
        verdict = "⚠  Minor impact — standard processing adequate."
    elif level == "MEDIUM":
        verdict = "⚠⚠ Moderate impact — correction recommended before mapping."
    else:
        verdict = "✘  SEVERE impact — correction REQUIRED before mineral mapping."
    impact.append(f"  {name:<40} → {verdict}")

impact.append("")
print("\n".join(impact))
report_lines.extend(impact)

# ── SAVE REPORT ───────────────────────────────────────────────────────────────
import os
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"  Report saved → {REPORT_PATH}")
