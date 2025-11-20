import torch
import brevitas.onnx as bo
import os
import json
import numpy as np

# --- UNIVERSAL IMPORTS ---
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType
)
from finn.builder.build_dataflow_steps import (
    step_qonnx_to_finn, step_tidy_up, step_streamline,
    step_convert_to_hw, step_create_dataflow_partition,
    step_specialize_layers, step_apply_folding_config,
    step_generate_estimate_reports
)

# --- USER CONFIGURATION ---
from model import FINNCompatibleGHM_MultiOutput

# Wrapper to force Input Quantization
from brevitas.quant import Int8ActPerTensorFloat
import torch.nn as nn
import brevitas.nn as qnn

class FinnInputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        # Force 8-bit signed input
        self.quant_input = qnn.QuantIdentity(
            bit_width=8,
            return_quant_tensor=True,
            act_quant=Int8ActPerTensorFloat
        )
        self.model = model

    def forward(self, x):
        return self.model(self.quant_input(x))

# ==============================================================================
# ⚡ TURBO MODE CONFIGURATION
# ==============================================================================
BUILD_DIR = "finn_build_universal"
ONNX_FILE = "temp_model.onnx"
TARGET_BOARD = "KV260_SOM"
TARGET_FPGA = "xck26-sfvc784-2LV-c"

# Aggressive Hardware Targets for KV260
TARGET_CLOCK_NS = 5.0    # 200 MHz (Standard is 10.0ns / 100MHz)
MAX_PE = 64              # Use more compute units (KV260 has ~1200 DSPs)
MAX_SIMD = 64            # Process larger vectors per cycle

# --- SETUP ---
if 'XILINX_VIVADO' not in os.environ:
    os.environ['XILINX_VIVADO'] = '/dummy/path'
os.environ['FINN_BUILD_DIR'] = os.path.abspath(BUILD_DIR)
os.makedirs(BUILD_DIR, exist_ok=True)

def get_largest_divisor(number, limit):
    """Finds the largest divisor of 'number' that is <= 'limit'."""
    for i in range(limit, 0, -1):
        if number % i == 0: return i
    return 1

# ==============================================================================
# 1. EXPORT
# ==============================================================================
print("--- 1. Exporting Model ---")

# Removed 'num_reg=2' because width/depth are now separate heads
raw_model = FINNCompatibleGHM_MultiOutput(in_channels=4, num_angles=6)
model = FinnInputWrapper(raw_model)
model.eval()

dummy_input = torch.randn(1, 4, 360, 640)
bo.export_qonnx(model, input_t=dummy_input, export_path=ONNX_FILE)
print(f"✅ Model exported to {ONNX_FILE}")

# ==============================================================================
# 2. AUTO-TUNE (TURBO)
# ==============================================================================
print("\n--- 2. Auto-Tuning Hardware Config ---")

# Load model
mw = ModelWrapper(ONNX_FILE)
cfg_temp = DataflowBuildConfig(output_dir=BUILD_DIR, board=TARGET_BOARD, fpga_part=TARGET_FPGA)

# Run transforms to identify nodes
mw = step_qonnx_to_finn(mw, cfg_temp)
mw = step_tidy_up(mw, cfg_temp)
mw = step_streamline(mw, cfg_temp)
mw = step_convert_to_hw(mw, cfg_temp)

folding_config = { "Defaults": { "ram_style": "auto" } }
mvau_nodes = [n for n in mw.graph.node if n.op_type.startswith("MVAU")]

print(f"   > Found {len(mvau_nodes)} compute layers. Optimizing for 200MHz...")

for i, node in enumerate(mvau_nodes):
    # Force name to match final build naming convention (MVAU_hls_0, 1, etc.)
    target_name = f"MVAU_hls_{i}"

    w_name = node.input[1]
    W = mw.get_initializer(w_name)

    if W is None: continue

    # Shapes: (InputVector, OutputChannels) for MVAU
    vec_len = W.shape[0] # Input Volume (determines SIMD)
    out_ch = W.shape[1]  # Output Channels (determines PE)

    # Optimization Math
    best_pe = get_largest_divisor(out_ch, MAX_PE)
    best_simd = get_largest_divisor(vec_len, MAX_SIMD)

    print(f"     [{target_name}] Shapes:{W.shape} -> PE:{best_pe:<2} SIMD:{best_simd:<2} | (OutCh:{out_ch}, Vec:{vec_len})")

    folding_config[target_name] = {
        "PE": int(best_pe),
        "SIMD": int(best_simd),
        "mem_mode": "internal_decoupled"
    }

config_path = os.path.join(BUILD_DIR, "auto_config.json")
with open(config_path, "w") as f:
    json.dump(folding_config, f, indent=2)

print(f"✅ Config saved to {config_path}")

# ==============================================================================
# 3. ESTIMATE
# ==============================================================================
print("\n--- 3. Running Estimation ---")

cfg = DataflowBuildConfig(
    output_dir = BUILD_DIR,
    fpga_part = TARGET_FPGA,
    board = TARGET_BOARD,
    synth_clk_period_ns = TARGET_CLOCK_NS,
    shell_flow_type = ShellFlowType.VIVADO_ZYNQ,
    generate_outputs = [DataflowOutputType.ESTIMATE_REPORTS],
    folding_config_file = config_path,
    steps = [
        step_qonnx_to_finn, step_tidy_up, step_streamline,
        step_convert_to_hw, step_create_dataflow_partition,
        step_specialize_layers, step_apply_folding_config,
        step_generate_estimate_reports
    ]
)

try:
    build_dataflow_cfg(ONNX_FILE, cfg=cfg)

    report_path = os.path.join(BUILD_DIR, "report", "estimate_network_performance.json")
    if not os.path.exists(report_path):
        report_path = os.path.join(BUILD_DIR, "intermediate_models", "step_generate_estimate_reports", "estimate_network_performance.json")

    if os.path.exists(report_path):
        with open(report_path, 'r') as f: res = json.load(f)
        print("\n" + "="*40)
        print("       UNIVERSAL TUNER RESULTS")
        print("="*40)
        print(f"🚀 FPS: {res.get('estimated_throughput_fps', 0):.2f}")
        print(f"🐢 Slowest: {res.get('max_cycles_node_name', 'Unknown')}")
        print("="*40)
    else:
        print("\n❌ Report file not generated.")

except Exception as e:
    print(f"\n❌ Build Error: {e}")