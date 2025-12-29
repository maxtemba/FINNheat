import os
import json
import torch
import brevitas.onnx as bo

from qonnx.core.modelwrapper import ModelWrapper
from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig, DataflowOutputType, ShellFlowType
)
from finn.builder.build_dataflow_steps import (
    step_qonnx_to_finn, step_tidy_up, step_streamline,
    step_convert_to_hw, step_create_dataflow_partition,
    step_specialize_layers, step_apply_folding_config,
    step_generate_estimate_reports
)

# --- config for Kria KV260
TARGET_BOARD = "KV260_SOM"
TARGET_FPGA = "xck26-sfvc784-2LV-c"
TARGET_CLOCK = 5.0  # equals 200 MHz frequency
MAX_PE = 64         # TODO
MAX_SIMD = 64       # TODO

# for partial FINN installation without Vivado refer to dummy path.
if 'XILINX_VIVADO' not in os.environ:
    os.environ['XILINX_VIVADO'] = '/dummy'


def export_to_qonnx(model, filename):
    """
    exports a model to Brevitas QNNX. Used by hardware estimator and export script.

    :param model: trained model.
    :param filename: output filename.
    :return: true if successfully exported, false otherwise.
    """
    model.eval()

    # dummy input image (Batch, Channels, Height, Width) to determine image flow through the network.
    dummy_input = torch.randn(1, 4, 360, 640)

    print(f"exporting QONNX model to: {filename}")
    try:
        bo.export_qonnx(model, input_t=dummy_input, export_path=filename)
        return True
    except Exception as e:
        print(f"export failed: {e}")
        return False

def get_folding_factor(total_channels, hardware_limit):
    """
    calculates the optimal hardware parallelism.

    :param total_channels: dimension of the weight tensor to parallelize.
    :param hardware_limit: maximum allowed parallelism factor.
    :return: optimal folding factor.
    """

    # largest divisor of (total_channels) that fits within the (hardware_limit).
    for i in range(hardware_limit, 0, -1):
        if total_channels % i == 0:
            return i
    return 1

def estimate_performance(model, build_name="finn_eval"):
    """
    function that runs the FINN compiler to estimate: (FPS) and resource usage (LUTs, BRAMs).
    based on the estimation pipeline:
    1. export.
    2. transform.
    3. auto folding.
    4. report.

    :param model: pytorch model.
    :param build_name: build directory.
    :return: dictionary containing hardware metrics.
    """

    # --- setup build directory structure
    build_dir = os.path.abspath(f"build_{build_name}")
    onnx_file = os.path.join(build_dir, "model.onnx")
    os.environ['FINN_BUILD_DIR'] = build_dir
    os.makedirs(build_dir, exist_ok=True)

    print(f"starting hardware estimation for: {build_name}")

    # 1. export model to QNNX

    if not export_to_qonnx(model, onnx_file):
        return None


    # 2. transform to hardware representation for later synthetase

    # loads model into Finn internal wrapper.
    mw = ModelWrapper(onnx_file)
    cfg_temp = DataflowBuildConfig(
        output_dir=build_dir,
        board=TARGET_BOARD,
        fpga_part=TARGET_FPGA,
        synth_clk_period_ns=TARGET_CLOCK
    )

    # runs FINN pipeline transformation.
    try:
        mw = step_qonnx_to_finn(mw, cfg_temp)
        mw = step_tidy_up(mw, cfg_temp)
        mw = step_streamline(mw, cfg_temp)
        mw = step_convert_to_hw(mw, cfg_temp)
    except Exception as e:
        print(f"graph preparation failed: {e}")
        return None


    # 3. auto folding for parallelism

    folding_config = {"Defaults": {"ram_style": "auto"}}

    # find nodes from the model.
    mvau_nodes = [n for n in mw.graph.node if n.op_type.startswith("MVAU")] # filters out non conv. and linear/dense layers
    print(f"found {len(mvau_nodes)} layers to accelerate.")

    # loop through filtered nodes/layers.
    for i, node in enumerate(mvau_nodes):

        # manually construct node names to avoid default unstable or empty node names.
        target_name = f"MVAU_hls_{i}"

        # get weight tensor to determine dimensions (kernel values).
        weights = mw.get_initializer(node.input[1])
        if weights is None: continue

        # optimal parallelism based on weight shapes.
        pe = get_folding_factor(weights.shape[1], MAX_PE)  # PE (Parallel Elements) corresponds to output channels
        simd = get_folding_factor(weights.shape[0], MAX_SIMD)   # SIMD (Single Instruction Multiple Data) corresponds to input channels

        folding_config[target_name] = {
            "PE": int(pe),
            "SIMD": int(simd),
            "mem_mode": "internal_decoupled"
        }

    # save folding configuration to JSON.
    config_path = os.path.join(build_dir, "auto_config.json")
    with open(config_path, "w") as f:
        json.dump(folding_config, f, indent=2)


    # 4. evaluation

    # custom pipeline setup for estimates.
    cfg = DataflowBuildConfig(
        output_dir=build_dir,
        board=TARGET_BOARD,
        fpga_part=TARGET_FPGA,
        synth_clk_period_ns=TARGET_CLOCK,
        folding_config_file=config_path,
        generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS],
        steps=[
            step_qonnx_to_finn, step_tidy_up, step_streamline,
            step_convert_to_hw, step_create_dataflow_partition,
            step_specialize_layers, step_apply_folding_config,
            step_generate_estimate_reports
        ]
    )

    # read report
    try:
        build_dataflow_cfg(onnx_file, cfg=cfg)

        report_path = os.path.join(build_dir, "report", "estimate_network_performance.json")

        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                res = json.load(f)

            return {
                "fps": res.get("estimated_throughput_fps", 0),
                "latency": res.get("estimated_latency_cycles", 0),
                "lut": res.get("total_luts", 0),
                "bram": res.get("total_brams", 0),
                "dsp": res.get("total_dsps", 0)
            }

        print(f"report missing at: {report_path}")
        return None

    except Exception as e:
        print(f"estimation process failed: {e}")
        return None