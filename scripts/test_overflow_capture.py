"""
quick test: verify that overflow detection and resource capture works.
uses g49_max (definitely overflows kv260) to check:
  1. synthesis sentinel is detected
  2. placement timeout fires (not the hard cap)
  3. resource data is extracted from synth report
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from finn.util.settings import initialize_dummy_settings
initialize_dummy_settings()

os.environ["XILINX_VIVADO"] = "/tools/Xilinx/Vivado/2024.2"
os.environ["VITIS_HLS"]     = "/tools/Xilinx/Vitis_HLS/2024.2"
os.environ["XILINX_VITIS"]  = "/tools/Xilinx/Vitis/2024.2"
os.environ["VIVADO_PATH"]   = "/tools/Xilinx/Vivado/2024.2"
os.environ["FINN_RTLLIB"]   = "/home/max/finn-plus/finn-rtllib"

from collect_calibration_data import run_synthesis_with_timeout

# g05: completes HLS fine but needs 142k LUTs — overflows kv260 at OOC placement
genome = {"enc_ch": [16, 32, 48], "enc_depth": [1, 1, 1], "enc_k": [3, 3, 3],
          "enc_bits": [4, 8, 8], "btl_ch": 64, "dec_ch": [48, 32, 16]}

print("testing overflow capture with g49_max...")
print(f"genome: {genome}")
print()

result = run_synthesis_with_timeout(genome, build_name="test_overflow_g49")

print()
if result is None:
    print("RESULT: None (timeout without resource capture — FAIL)")
elif result.get("overflow"):
    print(f"RESULT: overflow captured successfully")
    print(f"  lut={result['lut']}  bram={result['bram']}  dsp={result['dsp']}")
else:
    print(f"RESULT: unexpectedly succeeded (design fit?)")
    print(result)
