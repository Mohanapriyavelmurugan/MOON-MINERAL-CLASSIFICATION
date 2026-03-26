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

def banner(msg): 
    print("\n" + "#"*80)
    print(f"### {msg}")
    print("#"*80 + "\n")

if len(sys.argv) < 2:
    print("Usage: python RUN_M3_PIPELINE.py <path/to/scene.HDR>")
    # If no argument is provided, ask the user to input the path directly here,
    # or you can fall back to a default testing path.
    user_input = input("Please enter the path to the M3 SCENE HDR file: ")
    if user_input.strip() == "":
        print("No path provided. Exiting.")
        sys.exit(1)
    else:
        SCENE_HDR = os.path.abspath(user_input.strip('\"\''))
else:
    SCENE_HDR = os.path.abspath(sys.argv[1])

if not os.path.exists(SCENE_HDR):
    print(f"ERROR: Input file not found -> {SCENE_HDR}")
    sys.exit(1)

# Define the workspace as the directory where this script sits
WORKSPACE = os.path.dirname(os.path.abspath(__file__))

# The 4 stages of the pipeline
SCRIPTS = [
    ("1. Noise Characterisation", "MOON_NOISE_CHARACTERISATION.py"),
    ("2. Physics-Based Corrections", "MOON_PHYSICS_CORRECTIONS.py"),
    ("3. ML Denoising (CNN + Autoencoder)", "MOON_ML_DENOISING.py"),
    ("4. Scientific Validation", "MOON_VALIDATION.py"),
    ("5. Structure-Aware Contrastive Classification", "MOON_MINERAL_CLASSIFICATION.py")
]

banner("STARTING END-TO-END M3 PIPELINE")
print(f"Processing Scene: {SCENE_HDR}")

for stage_name, script_name in SCRIPTS:
    banner(stage_name)
    script_path = os.path.join(WORKSPACE, script_name)
    
    if not os.path.exists(script_path):
        print(f"ERROR: Script not found -> {script_path}")
        sys.exit(1)
        
    print(f"Running: python {script_name} \"{SCENE_HDR}\" ...\n")
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    process = subprocess.Popen(
        [sys.executable, script_path, SCENE_HDR], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=WORKSPACE
    )
    
    if process.stdout:
        for line in process.stdout:
            print(line, end="")
        
    process.wait()
    
    if process.returncode != 0:
        print(f"\n[!] ERROR in {stage_name}. Pipeline aborted.")
        sys.exit(1)

banner("PIPELINE COMPLETED SUCCESSFULLY")
print("The data is now ready for mineral identification.")

# Infer final output paths for the success message
scene_dir = os.path.dirname(SCENE_HDR)
scene_base = os.path.basename(SCENE_HDR).replace('.HDR', '').replace('.hdr', '')
final_hdr = os.path.join(scene_dir, "ML_Denoised", f"{scene_base}_FINAL.hdr")
val_txt = os.path.join(scene_dir, "Validation", "scientific_validation_proof.txt")

print(f"Final Output: {final_hdr}")
print(f"Validation Report: {val_txt}")
