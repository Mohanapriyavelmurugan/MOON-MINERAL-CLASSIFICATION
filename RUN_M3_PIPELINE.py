"""
================================================================================
RUN_M3_PIPELINE.py
Master execution script for the End-to-End M3 L2 Preprocessing Pipeline
================================================================================
Workflow:
  1. Noise Characterisation  -> measures raw noise levels
  2. Physics Corrections     -> fixes stripes, smile, thermal, saturation
  3. ML Denoising            -> applying spatial CNN (retrained) and 
                                Spectral Autoencoder (loaded if exists)
  4. Validation              -> generates statistical proof report
================================================================================
"""

import os
import subprocess
import sys

# Define the path to your scenes and scripts
WORKSPACE = r"d:\Moon"
SCENE_HDR = r"D:\Moon_Data\Scene_2\M3G20081201T064047_V01_RFL.HDR"

# The 4 stages of the pipeline
SCRIPTS = [
    ("1. Noise Characterisation", "MOON_NOISE_CHARACTERISATION.py"),
    ("2. Physics-Based Corrections", "MOON_PHYSICS_CORRECTIONS.py"),
    ("3. ML Denoising (CNN + Autoencoder)", "MOON_ML_DENOISING.py"),
    ("4. Scientific Validation", "MOON_VALIDATION.py")
]

def banner(msg): 
    print("\n" + "#"*80)
    print(f"### {msg}")
    print("#"*80 + "\n")

banner("STARTING END-TO-END M3 PIPELINE")
print(f"Processing Scene: {SCENE_HDR}")

for stage_name, script_name in SCRIPTS:
    banner(stage_name)
    script_path = os.path.join(WORKSPACE, script_name)
    
    if not os.path.exists(script_path):
        print(f"ERROR: Script not found -> {script_path}")
        sys.exit(1)
        
    print(f"Running: python {script_name} ...\n")
    
    # Run the script and stream the output to the console
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    process = subprocess.Popen(
        [sys.executable, script_path], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=WORKSPACE
    )
    
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    
    if process.returncode != 0:
        print(f"\n[!] ERROR in {stage_name}. Pipeline aborted.")
        sys.exit(1)

banner("PIPELINE COMPLETED SUCCESSFULLY")
print("The data is now ready for mineral identification.")
print("Final Output: D:\\Moon_Data\\Scene_2\\ML_Denoised\\M3G20081118T222604_V01_RFL_FINAL.hdr")
print("Validation Report: D:\\Moon_Data\\Scene_2\\Validation\\scientific_validation_proof.txt")
