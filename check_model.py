import torch
import brevitas.onnx as bo
import os
import json
import pprint
from model import FINNCompatibleGHM_MultiOutput  # Import your 3-output model

from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType
)

# --- Configuration ---
ONNX_MODEL_FILE = "trained_heatmap_model.onnx"
BUILD_DIR = "finn_build_reports_TRAINED"
MODEL_PATH = "trained_model.pth"
TARGET_BOARD = "Pynq-Z1"
TARGET_CLOCK_NS = 10.0

# === FIX FOR MACOS: Set dummy Vivado path to avoid reporting crash ===
if 'XILINX_VIVADO' not in os.environ:
    os.environ['XILINX_VIVADO'] = '/not/installed'
    print("⚠️  Running on macOS without Vivado - setting dummy path for reports")

os.environ['FINN_BUILD_DIR'] = os.path.abspath(BUILD_DIR)

# ==============================================================================
# STEP 1: FINN Compatibility Check (Export Trained Model)
# ==============================================================================
print("--- Starting Step 1: Exporting TRAINED Model ---")

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    print("Please run 'train.py' first.")
    exit()

model = FINNCompatibleGHM_MultiOutput()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 4, 360, 640)

try:
    bo.export_qonnx(
        model,
        input_t=dummy_input,
        export_path=ONNX_MODEL_FILE
    )
    print(f"✅ SUCCESS: Trained model exported to {ONNX_MODEL_FILE}")

except Exception as e:
    print(f"❌ FAILURE: Model export failed.")
    print("Error:", e)
    exit()

# ==============================================================================
# STEP 2: Get Analytical (Fast) Latency and Resource Predictions
# ==============================================================================
print("\n--- Starting Step 2: Generating Analytical (No HLS) Reports ---")

os.makedirs(BUILD_DIR, exist_ok=True)

cfg = DataflowBuildConfig(
    output_dir = BUILD_DIR,
    board = TARGET_BOARD,
    synth_clk_period_ns = TARGET_CLOCK_NS,
    generate_outputs = [
        DataflowOutputType.ESTIMATE_REPORTS
    ],

    # --- THIS IS THE FIX ---
    # 1. Tell the builder to optimize for 100 FPS. This enables folding.
    target_fps = 100
    # 2. The 'stop_after_step' line has been REMOVED.
    # --- END OF FIX ---
)

# 3. Run the FINN+ builder
try:
    build_dataflow_cfg(
        ONNX_MODEL_FILE,
        cfg = cfg
    )

    print("✅ SUCCESS: Analytical reports generated!")

    report_file = os.path.join(BUILD_DIR, "report", "estimate_network_performance.json")

    try:
        with open(report_file, 'r') as f:
            predictions = json.load(f)

        print("\n" + "="*50)
        print("   ANALYTICAL PREDICTIONS FOR TRAINED MODEL (NO HLS)")
        print("="*50)
        pprint.pprint(predictions)
        print("="*50)

    except FileNotFoundError:
        print(f"❌ Error: Report file not found at {report_file}")

except Exception as e:
    # This will likely catch the harmless 'FINN_RTLLIB' error
    print(f"\n⚠️  FINN build process exited with an error (this is expected on Mac).")
    print(f"   Error: {e}")
    print("   Checking if the report was generated anyway...")

    report_file = os.path.join(BUILD_DIR, "report", "estimate_network_performance.json")

    try:
        with open(report_file, 'r') as f:
            predictions = json.load(f)

        print("\n" + "="*50)
        print("   ANALYTICAL PREDICTIONS FOR TRAINED MODEL (NO HLS)")
        print("="*50)
        pprint.pprint(predictions)
        print("="*50)

    except FileNotFoundError:
        print(f"❌ Error: Report file not found at {report_file}")