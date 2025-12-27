import torch
import brevitas.onnx as bo
import os
import sys
import ast

# --- PATH SETUP ---
# Go up one level from 'tests/' to find 'core'
sys.path.append("..")

from core.models import NAS_GHM_Model

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Paths (Relative to 'tests/' folder)
GENOME_FILE = "best_genome.txt"
WEIGHTS_FILE = "trained_model_hggd.pth"
OUTPUT_ONNX = "nas_model_export.onnx"

# ==============================================================================
# MAIN EXPORT SCRIPT
# ==============================================================================
def main():
    print(f"🚀 Starting FINN/ONNX Export")

    # 1. Load Genome
    if not os.path.exists(GENOME_FILE):
        print(f"❌ Error: Genome file not found at {GENOME_FILE}"); return

    with open(GENOME_FILE, "r") as f:
        genome = ast.literal_eval(f.read().strip())
    print(f"🧬 Genome Loaded: {genome}")

    # 2. Build Model
    model = NAS_GHM_Model(genome)

    # 3. Load Weights
    if not os.path.exists(WEIGHTS_FILE):
        print(f"❌ Error: Weights file not found at {WEIGHTS_FILE}"); return

    # Load to CPU for export
    state_dict = torch.load(WEIGHTS_FILE, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print("⚖️  Weights Loaded Successfully")

    # 4. Prepare Dummy Input
    # Shape: [1, 4, 360, 640] (Batch, Channels, Height, Width)
    # Adjust resolution if your 'downsample_factor' was different!
    dummy_input = torch.randn(1, 4, 360, 640)

    # 5. Export to QONNX
    # We use export_qonnx (not standard torch.onnx) because it preserves
    # the quantization nodes required by FINN.
    print(f"📦 Exporting to {OUTPUT_ONNX}...")

    try:
        bo.export_qonnx(
            model,
            input_t=dummy_input,
            export_path=OUTPUT_ONNX
        )
        print(f"\n✅ SUCCESS! Model saved to: {os.path.abspath(OUTPUT_ONNX)}")
        print("   You can now copy this .onnx file to your Linux machine.")
        print("   Run the FINN Docker container and point the builder to this file.")

    except Exception as e:
        print(f"\n❌ Export Failed: {e}")

if __name__ == "__main__":
    main()