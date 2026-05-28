import os
import re
import glob
import json
import shutil
from contextlib import contextmanager

from finn.util.settings import initialize_dummy_settings
initialize_dummy_settings()

from core.export import export_to_qonnx
from config import TARGET_BOARD, TARGET_FPGA, TARGET_CLOCK_NS, BUILDS_DIR, FINN_TMP_DIR
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
    step_hw_codegen, step_hw_ipgen,
    step_create_stitched_ip, step_out_of_context_synthesis,
)


def _parse_synth_utilization_rpt():
    # reads vivado synthesis utilization report from FINN_TMP when ooc synthesis fails
    # synthesis (synth_1) always completes even on overflow, only impl_1 (place_design) fails
    # returns {lut, lut_ram, bram, dsp, overflow=True} or None if report missing
    pattern = os.path.join(
        FINN_TMP_DIR, "synth_out_of_context_*",
        "results_finn_design_wrapper", "vivadocompile",
        "vivadocompile.runs", "synth_1",
        "finn_design_wrapper_utilization_synth.rpt"
    )
    matches = glob.glob(pattern)
    if not matches:
        return None
    rpt_path = sorted(matches)[-1]  # most recent if multiple
    try:
        with open(rpt_path) as f:
            text = f.read()

        def parse_row(label):
            m = re.search(rf'\|\s+{re.escape(label)}\s*\*?\s+\|\s+(\d[\d,]*)', text)
            return int(m.group(1).replace(",", "")) if m else 0

        lut_logic = parse_row("LUT as Logic")
        lut_mem   = parse_row("LUT as Memory")
        ramb36    = parse_row("RAMB36/FIFO")
        ramb18    = parse_row("RAMB18")
        dsp       = parse_row("DSPs")
        print(f"  overflow synth report: lut={lut_logic+lut_mem}  bram={ramb36*2+ramb18}  dsp={dsp}")
        return {
            "lut":      lut_logic + lut_mem,
            "lut_ram":  lut_mem,
            "bram":     ramb36 * 2 + ramb18,
            "dsp":      dsp,
            "overflow": True,
        }
    except Exception as e:
        print(f"failed to parse utilization report: {e}")
        return None


def get_folding_factor(channels, limit):
    # largest divisor of channels that is <= limit
    for i in range(limit, 0, -1):
        if channels % i == 0: return i
    return 1


def build_folding_config(mw, config_path, parallelism=2):
    # write auto folding config for all MVAU layers
    # pe/simd = best divisor <= parallelism per layer
    folding = {"Defaults": {"ram_style": "auto"}}
    mvau_nodes = [n for n in mw.graph.node if n.op_type.startswith("MVAU")]

    for i, node in enumerate(mvau_nodes):
        name = f"MVAU_hls_{i}"
        w = mw.get_initializer(node.input[1])  # weights
        if w is None: continue

        fan_in       = w.shape[0]  # width controls SIMD (memory)
        out_channels = w.shape[1]  # height controls PE (compute)

        pe_target   = parallelism
        simd_target = parallelism

        pe   = get_folding_factor(out_channels, pe_target)
        simd = get_folding_factor(fan_in, simd_target)

        folding[name] = {"PE": int(pe), "SIMD": int(simd), "mem_mode": "internal_decoupled", "resType": "dsp"}

    with open(config_path, "w") as f: json.dump(folding, f, indent=2)


@contextmanager
def _finn_build_env(build_name, need_vivado=False):
    # set up build dir + env vars, clean up on exit
    # need_vivado=True: real synthesis path, creates FINN_TMP, sets FINN_CUSTOM_HLS
    # need_vivado=False: estimation only, falls back to a dummy XILINX_VIVADO path
    build_dir = os.path.join(BUILDS_DIR, f"build_{build_name}")
    os.environ['FINN_BUILD_DIR'] = build_dir
    os.makedirs(build_dir, exist_ok=True)

    if need_vivado:
        os.makedirs(FINN_TMP_DIR, exist_ok=True)
        # HLS TCL scripts require FINN_CUSTOM_HLS to be set (even if unused)
        os.environ.setdefault('FINN_CUSTOM_HLS', '')
    else:
        # allow estimation without a real vivado installation
        os.environ.setdefault('XILINX_VIVADO', '/dummy')

    try:
        yield build_dir
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)
        if need_vivado:
            shutil.rmtree(FINN_TMP_DIR, ignore_errors=True)


def _prepare_finn_ir(model, build_dir, parallelism=2):
    # export to qonnx, run ir conversion steps, write folding config
    # returns (onnx_file, config_path) or (None, None) on failure
    onnx_file = os.path.join(build_dir, "model.onnx")

    if not export_to_qonnx(model, onnx_file):
        return None, None

    mw = ModelWrapper(onnx_file)
    cfg_temp = DataflowBuildConfig(
        output_dir=build_dir,
        board=TARGET_BOARD,
        fpga_part=TARGET_FPGA,
        synth_clk_period_ns=TARGET_CLOCK_NS
    )

    try:
        mw = step_qonnx_to_finn(mw, cfg_temp)
        mw = step_tidy_up(mw, cfg_temp)
        mw = step_streamline(mw, cfg_temp)
        mw = step_convert_to_hw(mw, cfg_temp)
    except Exception as e:
        print(f"graph prep failed: {e}")
        return None, None

    config_path = os.path.join(build_dir, "auto_config.json")
    build_folding_config(mw, config_path, parallelism=parallelism)

    return onnx_file, config_path


def estimate_performance(model, build_name="finn_eval", parallelism=2):
    # fast finn estimate: fps, lut, bram, dsp. Returns metric dict or None
    print(f"starting hardware estimation for: {build_name}")

    with _finn_build_env(build_name, need_vivado=False) as build_dir:
        onnx_file, config_path = _prepare_finn_ir(model, build_dir, parallelism=parallelism)
        if not onnx_file:
            return None

        cfg = DataflowBuildConfig(
            output_dir=build_dir, board=TARGET_BOARD,
            fpga_part=TARGET_FPGA, synth_clk_period_ns=TARGET_CLOCK_NS,
            folding_config_file=config_path,
            generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS],
            steps=[
                step_qonnx_to_finn, step_tidy_up, step_streamline,
                step_convert_to_hw, step_create_dataflow_partition,
                step_specialize_layers, step_apply_folding_config,
                step_generate_estimate_reports
            ]
        )

        try:
            build_dataflow_cfg(onnx_file, cfg=cfg)
        except Exception as e:
            print(f"estimation failed: {e}")
            return None

        p_path = os.path.join(build_dir, "report", "estimate_network_performance.json")
        r_path = os.path.join(build_dir, "report", "estimate_layer_resources.json")

        res_perf = {}
        res_res  = {}
        if os.path.exists(p_path):
            with open(p_path, 'r') as f: res_perf = json.load(f)
        if os.path.exists(r_path):
            with open(r_path, 'r') as f: res_res  = json.load(f)

        total = res_res.get("total", {})
        return {
            "fps":     res_perf.get("estimated_throughput_fps", 0),
            "latency": res_perf.get("critical_path_cycles", 0),
            "lut":     total.get("LUT", 0),
            "bram":    total.get("BRAM_18K", 0),
            "dsp":     total.get("DSP", 0)
        }


def synthesize_performance(model, build_name="finn_synth", generate_bitfile=False, parallelism=2):
    # full hls + ooc synthesis, returns real synth metrics or None
    # generate_bitfile=True also runs full place-and-route
    print(f"starting full synthesis for: {build_name}")

    with _finn_build_env(build_name, need_vivado=True) as build_dir:
        onnx_file, config_path = _prepare_finn_ir(model, build_dir, parallelism=parallelism)
        if not onnx_file:
            return None

        outputs = [DataflowOutputType.ESTIMATE_REPORTS, DataflowOutputType.STITCHED_IP, DataflowOutputType.OOC_SYNTH]
        if generate_bitfile:
            outputs.append(DataflowOutputType.BITFILE)

        synth_steps = [
            step_qonnx_to_finn, step_tidy_up, step_streamline,
            step_convert_to_hw, step_create_dataflow_partition,
            step_specialize_layers, step_apply_folding_config,
            step_generate_estimate_reports,   # keep estimates for comparison
            # step_set_fifo_depths omitted, needs rtl sim and not required for ooc
            step_hw_codegen,                  # hls c++ codegen per layer
            step_hw_ipgen,                    # hls c-synthesis per layer
            step_create_stitched_ip,          # stitch hls ips into one design
            step_out_of_context_synthesis,    # vivado ooc for real resource numbers
        ]

        cfg = DataflowBuildConfig(
            output_dir=build_dir, board=TARGET_BOARD,
            fpga_part=TARGET_FPGA, synth_clk_period_ns=TARGET_CLOCK_NS,
            folding_config_file=config_path,
            generate_outputs=outputs,
            steps=synth_steps,
        )

        try:
            build_dataflow_cfg(onnx_file, cfg=cfg)
        except Exception as e:
            print(f"synthesis failed: {e}")
            return _parse_synth_utilization_rpt()  # parsed before FINN_TMP cleanup

        ooc_path   = os.path.join(build_dir, "report", "ooc_synth_and_timing.json")
        est_p_path = os.path.join(build_dir, "report", "estimate_network_performance.json")

        if not os.path.exists(ooc_path):
            print(f"synthesis report not found at: {ooc_path}")
            return _parse_synth_utilization_rpt()

        with open(ooc_path, 'r') as f:
            ooc = json.load(f)

        # ooc json is flat, no resources/timing nesting
        # estimated performance comes from the dataflow model, not affected by ooc
        est_fps = 0
        if os.path.exists(est_p_path):
            with open(est_p_path, 'r') as f:
                est_fps = json.load(f).get("estimated_throughput_fps", 0)

        wns  = ooc.get("WNS", None)
        fmax = ooc.get("fmax_mhz", None)
        clk  = round(1000.0 / fmax, 3) if fmax else TARGET_CLOCK_NS

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
