import sys
import os

# init finn+ settings before any finn imports
from finn.util.settings import initialize_dummy_settings
initialize_dummy_settings()

import config

# override dummy paths with real vivado/vitis installations
os.environ["XILINX_VIVADO"] = config.XILINX_VIVADO_PATH
os.environ["VITIS_HLS"]     = config.VITIS_HLS_PATH
os.environ["XILINX_VITIS"]  = config.XILINX_VITIS_PATH
os.environ["VIVADO_PATH"]   = config.XILINX_VIVADO_PATH
os.environ["FINN_RTLLIB"]   = config.FINN_RTLLIB_PATH

from core.models import load_nas_model
from core.hardware import synthesize_performance

BUILD_NAME       = "synth_best_genome"
GENERATE_BITFILE = False  # set True to also run full place-and-route


def main():
    print("starting full hardware synthesis for the best genome...")

    try:
        model = load_nas_model(config.ACTIVE_GENOME, device='cpu')
        print("model built successfully.")
    except Exception as e:
        print(f"setup failed: {e}")
        return

    print(f"running FINN full synthesis (GENERATE_BITFILE={GENERATE_BITFILE})...")
    print("note: takes significantly longer than estimation (30-120+ min).\n")

    metrics = synthesize_performance(model, build_name=BUILD_NAME, generate_bitfile=GENERATE_BITFILE)

    if metrics:
        wns      = metrics['timing_wns_ns']
        clk      = metrics['clk_period_ns']
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

        lut_pct  = metrics['lut']  / config.HARDWARE_LIMITS["LUT"]  * 100
        bram_pct = metrics['bram'] / config.HARDWARE_LIMITS["BRAM"] * 100
        dsp_pct  = metrics['dsp']  / config.HARDWARE_LIMITS["DSP"]  * 100

        print(f"\nKV260 utilization:")
        print(f"  LUTs:  {lut_pct:.1f}%  ({metrics['lut']} / {config.HARDWARE_LIMITS['LUT']})")
        print(f"  BRAMs: {bram_pct:.1f}%  ({metrics['bram']} / {config.HARDWARE_LIMITS['BRAM']})")
        print(f"  DSPs:  {dsp_pct:.1f}%  ({metrics['dsp']} / {config.HARDWARE_LIMITS['DSP']})")

        fits = lut_pct <= 100 and bram_pct <= 100 and dsp_pct <= 100 and timing_ok
        print(f"\nFits on KV260: {'YES' if fits else 'NO'}")
    else:
        print("synthesis failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
