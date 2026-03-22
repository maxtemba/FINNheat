import sys
import os
import json

# Initialize FINN+ settings before any FINN imports
from finn.util.settings import initialize_dummy_settings
initialize_dummy_settings()

# override dummy paths with real vivado/vitis installations
os.environ["XILINX_VIVADO"] = "/tools/Xilinx/Vivado/2024.2"
os.environ["VITIS_HLS"]     = "/tools/Xilinx/Vitis_HLS/2024.2"
os.environ["XILINX_VITIS"]  = "/tools/Xilinx/Vitis/2024.2"
os.environ["VIVADO_PATH"]   = "/tools/Xilinx/Vivado/2024.2"
os.environ["FINN_RTLLIB"]   = "/home/max/finn-plus/finn-rtllib"

sys.path.append("..")
from core.utils import load_nas_model
from core.hardware import export_to_qonnx, build_folding_config, TARGET_BOARD, TARGET_FPGA, TARGET_CLOCK

from qonnx.core.modelwrapper import ModelWrapper
from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig, DataflowOutputType
)
from finn.builder.build_dataflow_steps import (
    step_qonnx_to_finn, step_tidy_up, step_streamline,
    step_convert_to_hw, step_create_dataflow_partition,
    step_specialize_layers, step_apply_folding_config,
    step_generate_estimate_reports,
    step_hw_codegen,
    step_hw_ipgen,
    step_create_stitched_ip,
    step_out_of_context_synthesis,
)

# --- config paths
GENOME_FILE = "best_genome.txt"
BUILD_NAME  = "synth_best_genome"

# set to True to also run full place-and-route and generate a bitfile
GENERATE_BITFILE = False


def synthesize_performance(model, build_name="finn_synth"):
    """
    run full HLS + OOC synthesis (optionally bitfile) for a model using FINN compiler.
    1. export model to QONNX.
    2. prepare FINN IR + auto folding (same as estimate_performance).
    3. run HLS IP generation and stitch IP.
    4. run out-of-context synthesis to get real resource numbers.
    5. optionally run full place-and-route to generate a bitfile.

    :param model: pytorch model.
    :param build_name: FINN build name.
    :return: dict with real synthesis metrics, or None on failure.
    """

    build_dir = os.path.abspath(f"build_{build_name}")
    onnx_file = os.path.join(build_dir, "model.onnx")
    os.environ['FINN_BUILD_DIR'] = build_dir
    os.makedirs(build_dir, exist_ok=True)

    # FINN internally also needs its own tmp dir to exist
    finn_tmp = os.path.join(os.path.dirname(build_dir), "FINN_TMP")
    os.makedirs(finn_tmp, exist_ok=True)

    # HLS TCL scripts require FINN_CUSTOM_HLS to be set (even if unused)
    if 'FINN_CUSTOM_HLS' not in os.environ:
        os.environ['FINN_CUSTOM_HLS'] = ''

    print(f"starting full synthesis for: {build_name}")

    if not export_to_qonnx(model, onnx_file):
        return None

    # --- auto folding: prepare FINN IR, then build config via core/hardware.py ---
    mw = ModelWrapper(onnx_file)
    cfg_temp = DataflowBuildConfig(
        output_dir=build_dir,
        board=TARGET_BOARD,
        fpga_part=TARGET_FPGA,
        synth_clk_period_ns=TARGET_CLOCK
    )

    try:
        mw = step_qonnx_to_finn(mw, cfg_temp)
        mw = step_tidy_up(mw, cfg_temp)
        mw = step_streamline(mw, cfg_temp)
        mw = step_convert_to_hw(mw, cfg_temp)
    except Exception as e:
        print(f"graph prep failed: {e}")
        return None

    config_path = os.path.join(build_dir, "auto_config.json")
    build_folding_config(mw, config_path)

    # --- synthesis output targets ---
    outputs = [DataflowOutputType.STITCHED_IP, DataflowOutputType.OOC_SYNTH]
    if GENERATE_BITFILE:
        outputs.append(DataflowOutputType.BITFILE)

    # --- full synthesis build steps ---
    synth_steps = [
        step_qonnx_to_finn,
        step_tidy_up,
        step_streamline,
        step_convert_to_hw,
        step_create_dataflow_partition,
        step_specialize_layers,
        step_apply_folding_config,
        step_generate_estimate_reports,   # keep estimates for comparison
        # step_set_fifo_depths omitted: requires RTL sim, not needed for OOC synth
        step_hw_codegen,                  # generate HLS C++ code for every layer
        step_hw_ipgen,                    # HLS C-synthesis for every layer
        step_create_stitched_ip,          # stitch all HLS IPs into one design
        step_out_of_context_synthesis,    # run Vivado OOC for real resource numbers
    ]

    cfg = DataflowBuildConfig(
        output_dir=build_dir,
        board=TARGET_BOARD,
        fpga_part=TARGET_FPGA,
        synth_clk_period_ns=TARGET_CLOCK,
        folding_config_file=config_path,
        generate_outputs=outputs,
        steps=synth_steps,
    )

    try:
        build_dataflow_cfg(onnx_file, cfg=cfg)

        # --- read synthesis results ---
        ooc_path  = os.path.join(build_dir, "report", "ooc_synth_and_timing.json")
        est_p_path = os.path.join(build_dir, "report", "estimate_network_performance.json")

        if not os.path.exists(ooc_path):
            print(f"synthesis report not found at: {ooc_path}")
            return None

        with open(ooc_path, 'r') as f:
            ooc = json.load(f)

        # ooc json is flat (no "resources"/"timing" nesting)
        # estimated performance is still from the dataflow model (not affected by OOC)
        est_fps = 0
        if os.path.exists(est_p_path):
            with open(est_p_path, 'r') as f:
                est_fps = json.load(f).get("estimated_throughput_fps", 0)

        wns = ooc.get("WNS", None)
        fmax = ooc.get("fmax_mhz", None)
        clk  = round(1000.0 / fmax, 3) if fmax else TARGET_CLOCK

        return {
            "fps_estimate":  est_fps,
            "lut":           ooc.get("LUT", 0),
            "lut_ram":       ooc.get("LUTRAM", 0),
            "ff":            ooc.get("FF", 0),
            "bram":          int(ooc.get("BRAM_18K", 0)) + 2 * int(ooc.get("BRAM_36K", 0)),
            "dsp":           ooc.get("DSP", 0),
            "timing_wns_ns": wns,
            "clk_period_ns": clk,
        }

    except Exception as e:
        print(f"synthesis failed: {e}")
        return None


def main():
    print("starting full hardware synthesis for the best genome...")

    try:
        model = load_nas_model(GENOME_FILE, device='cpu')
        print("model built successfully.")
    except Exception as e:
        print(f"setup failed: {e}")
        return

    print(f"running FINN full synthesis (GENERATE_BITFILE={GENERATE_BITFILE})...")
    print("note: this will take significantly longer than estimation (30–120+ min).\n")

    metrics = synthesize_performance(model, build_name=BUILD_NAME)

    if metrics:
        wns = metrics['timing_wns_ns']
        clk = metrics['clk_period_ns']
        freq_mhz = round(1000.0 / clk, 1) if clk else "?"
        timing_ok = wns is not None and wns >= 0

        print("\nFull Synthesis Results (OOC):")
        print("------------------------------")
        print(f"FPS estimate:  {metrics['fps_estimate']:.2f}")
        print(f"LUTs:          {metrics['lut']}")
        print(f"LUT RAMs:      {metrics['lut_ram']}")
        print(f"FFs:           {metrics['ff']}")
        print(f"BRAMs (18K):   {metrics['bram']}")
        print(f"DSPs:          {metrics['dsp']}")
        print(f"Timing:        {freq_mhz} MHz  (WNS = {wns} ns)  {'PASS' if timing_ok else 'FAIL'}")
        print("------------------------------")

        # KV260 resource limits
        LUT_MAX  = 117120
        BRAM_MAX = 288
        DSP_MAX  = 1248

        lut_pct  = metrics['lut']  / LUT_MAX  * 100
        bram_pct = metrics['bram'] / BRAM_MAX * 100
        dsp_pct  = metrics['dsp']  / DSP_MAX  * 100

        print(f"\nKV260 utilization:")
        print(f"  LUTs:  {lut_pct:.1f}%  ({metrics['lut']} / {LUT_MAX})")
        print(f"  BRAMs: {bram_pct:.1f}%  ({metrics['bram']} / {BRAM_MAX})")
        print(f"  DSPs:  {dsp_pct:.1f}%  ({metrics['dsp']} / {DSP_MAX})")

        fits = lut_pct <= 100 and bram_pct <= 100 and dsp_pct <= 100 and timing_ok
        print(f"\nFits on KV260: {'YES' if fits else 'NO'}")
    else:
        print("synthesis failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
