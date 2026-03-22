import sys
import os

# Initialize FINN+ settings before any FINN imports
from finn.util.settings import initialize_dummy_settings
initialize_dummy_settings()

# override dummy paths with real vivado/vitis installations
os.environ["XILINX_VIVADO"] = "/tools/Xilinx/Vivado/2024.2"
os.environ["VITIS_HLS"]     = "/tools/Xilinx/Vitis_HLS/2024.2"
os.environ["XILINX_VITIS"]  = "/tools/Xilinx/Vitis/2024.2"
os.environ["VIVADO_PATH"]   = "/tools/Xilinx/Vivado/2024.2"
os.environ["FINN_RTLLIB"]   = "/home/max/finn-plus/finn-rtllib"

from core.models import load_nas_model
from core.hardware import synthesize_performance

# --- config
GENOME_FILE      = "../genomes/best_genome.txt"
BUILD_NAME       = "synth_best_genome"
GENERATE_BITFILE = False  # set to True to also run full place-and-route


def main():
    print("starting full hardware synthesis for the best genome...")

    try:
        model = load_nas_model(GENOME_FILE, device='cpu')
        print("model built successfully.")
    except Exception as e:
        print(f"setup failed: {e}")
        return

    print(f"running FINN full synthesis (GENERATE_BITFILE={GENERATE_BITFILE})...")
    print("note: this will take significantly longer than estimation (30-120+ min).\n")

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
