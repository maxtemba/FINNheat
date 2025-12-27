import torch
import torch.nn as nn
import brevitas.onnx as bo
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
import os
import json
import numpy as np

# --- UNIVERSAL IMPORTS ---
from qonnx.core.modelwrapper import ModelWrapper
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

# --- HARDWARE CONSTANTS (KV260) ---
TARGET_BOARD = "KV260_SOM"
TARGET_FPGA = "xck26-sfvc784-2LV-c"
TARGET_CLOCK_NS = 5.0    # 200 MHz
MAX_PE = 64              # Max Compute Units
MAX_SIMD = 64            # Max Vector Width

# --- SETUP ENV ---
# Prevent crash if Vivado isn't found (Mac/Standard PC)
if 'XILINX_VIVADO' not in os.environ:
    os.environ['XILINX_VIVADO'] = '/dummy/path'

# ==============================================================================
# 1. HELPER CLASSES
# ==============================================================================
class FinnInputWrapper(nn.Module):
    """Wraps any model to force 8-bit quantization at the input."""
    def __init__(self, model):
        super().__init__()
        self.quant_input = qnn.QuantIdentity(
            bit_width=8, return_quant_tensor=True, act_quant=Int8ActPerTensorFloat
        )
        self.model = model

    def forward(self, x):
        return self.model(self.quant_input(x))

def get_largest_divisor(number, limit):
    """Finds the largest divisor of 'number' that is <= 'limit'."""
    for i in range(limit, 0, -1):
        if number % i == 0: return i
    return 1

# ==============================================================================
# 2. THE ESTIMATOR FUNCTION
# ==============================================================================
def estimate_performance(model, build_name="finn_eval"):
    """
    Compiles a PyTorch model to FINN intermediate representation and estimates
    performance (FPS, LUTs, BRAMs) on the KV260.

    Args:
        model (nn.Module): The PyTorch model to check.
        build_name (str): Unique name for the build directory.

    Returns:
        dict: { 'fps': float, 'lut': int, 'bram': int, 'dsp': int }
        or None if build fails.
    """

    BUILD_DIR = f"build_{build_name}"
    ONNX_FILE = f"{BUILD_DIR}/model.onnx"
    os.environ['FINN_BUILD_DIR'] = os.path.abspath(BUILD_DIR)
    os.makedirs(BUILD_DIR, exist_ok=True)

    print(f"   ⚙️ Hardware Check: '{build_name}'...")

    # --- A. Export to ONNX ---
    # Wrap model if it doesn't already have the input quantizer
    # (NAS models usually have it, but safety first)
    if not hasattr(model, 'quant_input'):
        model = FinnInputWrapper(model)

    model.eval()
    # Dummy input shape: [Batch, Channels, Height, Width]
    dummy_input = torch.randn(1, 4, 360, 640)

    try:
        bo.export_qonnx(model, input_t=dummy_input, export_path=ONNX_FILE)
    except Exception as e:
        print(f"      ❌ Export Failed: {e}")
        return None

    # --- B. Auto-Tune (Calculate Folding) ---
    mw = ModelWrapper(ONNX_FILE)
    cfg_temp = DataflowBuildConfig(output_dir=BUILD_DIR, board=TARGET_BOARD, fpga_part=TARGET_FPGA)

    # Run prep steps to get graph structure
    try:
        mw = step_qonnx_to_finn(mw, cfg_temp)
        mw = step_tidy_up(mw, cfg_temp)
        mw = step_streamline(mw, cfg_temp)
        mw = step_convert_to_hw(mw, cfg_temp)
    except Exception as e:
        print(f"      ❌ Graph Prep Failed: {e}")
        return None

    folding_config = { "Defaults": { "ram_style": "auto" } }
    mvau_nodes = [n for n in mw.graph.node if n.op_type.startswith("MVAU")]

    # Calculate Parallelism for every layer
    for i, node in enumerate(mvau_nodes):
        target_name = f"MVAU_hls_{i}"
        w_name = node.input[1]
        W = mw.get_initializer(w_name)
        if W is None: continue

        vec_len = W.shape[0] # Input Vector
        out_ch = W.shape[1]  # Output Channels

        best_pe = get_largest_divisor(out_ch, MAX_PE)
        best_simd = get_largest_divisor(vec_len, MAX_SIMD)

        folding_config[target_name] = {
            "PE": int(best_pe),
            "SIMD": int(best_simd),
            "mem_mode": "internal_decoupled"
        }

    config_path = os.path.join(BUILD_DIR, "auto_config.json")
    with open(config_path, "w") as f:
        json.dump(folding_config, f, indent=2)

    # --- C. Run Estimation ---
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

        # Parse Report
        report_path = os.path.join(BUILD_DIR, "report", "estimate_network_performance.json")
        if not os.path.exists(report_path):
            report_path = os.path.join(BUILD_DIR, "intermediate_models", "step_generate_estimate_reports", "estimate_network_performance.json")

        if os.path.exists(report_path):
            with open(report_path, 'r') as f: res = json.load(f)

            metrics = {
                "fps": res.get("estimated_throughput_fps", 0),
                "latency": res.get("estimated_latency_cycles", 0),
                "lut": res.get("total_luts", 0),
                "bram": res.get("total_brams", 0),
                "dsp": res.get("total_dsps", 0)
            }
            return metrics
        else:
            print("      ⚠️ Report not generated (Likely Vivado missing). Using fallback.")
            return None

    except Exception as e:
        print(f"      ❌ Build Error: {e}")
        return None