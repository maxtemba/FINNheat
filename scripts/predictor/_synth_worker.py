# synthesis worker, called as a subprocess by collect_calibration_data.py
# takes genome as json arg, writes result to output json
import sys
import os
import json
import ast

# clear stale FINN_BUILD_DIR inherited from parent before any finn imports
os.environ.pop("FINN_BUILD_DIR", None)

# init finn settings before any finn imports
from finn.util.settings import initialize_dummy_settings
initialize_dummy_settings()

import config

os.environ["XILINX_VIVADO"] = config.XILINX_VIVADO_PATH
os.environ["VITIS_HLS"]     = config.VITIS_HLS_PATH
os.environ["XILINX_VITIS"]  = config.XILINX_VITIS_PATH
os.environ["VIVADO_PATH"]   = config.XILINX_VIVADO_PATH
os.environ["FINN_RTLLIB"]   = config.FINN_RTLLIB_PATH

from core.models import NAS_GHM_Model
from core.hardware import synthesize_performance

if __name__ == "__main__":
    # args: genome_json build_name output_json [parallelism]
    genome      = ast.literal_eval(sys.argv[1])
    build_name  = sys.argv[2]
    out_path    = sys.argv[3]
    parallelism = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    model  = NAS_GHM_Model(genome).cpu()
    result = synthesize_performance(model, build_name=build_name, generate_bitfile=False, parallelism=parallelism)

    with open(out_path, "w") as f:
        json.dump(result, f)
