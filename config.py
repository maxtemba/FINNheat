# central config for the finn-heat pipeline
# paths resolve from project root so callers don't depend on cwd
# import what you need, do not redefine constants in scripts
import os

# config.py lives at the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _p(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


# data
DATA_DIR      = _p("data", "graspnet")
IMG_CACHE_DIR = _p("data", "graspnet_img_cache")
CAMERA        = "kinect"
DOWNSAMPLE    = 8

# artifacts
GENOMES_DIR    = _p("genomes")
OUTPUTS_DIR    = _p("scripts", "outputs")
PREDICTORS_DIR = _p("scripts", "outputs", "predictors")
RESULTS_DIR    = _p("results")
BUILDS_DIR     = _p("builds")
FINN_TMP_DIR   = _p("FINN_TMP")

# active genome / weights used by training, export, synthesis, verification
ACTIVE_GENOME  = _p("scripts", "outputs", "gen33_genome.txt")
ACTIVE_WEIGHTS = _p("scripts", "outputs", "gen33_full_model.pth")

# nas writes its best genome here
BEST_NAS_GENOME = _p("genomes", "best_nas_genome.txt")

# live logs, overwritten on each run
NAS_LOG      = _p("results", "nas_search_log.csv")
TRAINING_LOG = _p("results", "training_log.csv")


# predictor pickles, one per parallelism level
def predictor_path(parallelism):
    return os.path.join(PREDICTORS_DIR, f"hw_predictor_p{parallelism}.pkl")


# calibration data dumps used to train the predictors
def calibration_data_path(parallelism):
    return os.path.join(OUTPUTS_DIR, f"calibration_data_p{parallelism}.json")


# hardware target (kria kv260)
TARGET_BOARD    = "KV260_SOM"
TARGET_FPGA     = "xck26-sfvc784-2LV-c"
TARGET_CLOCK_NS = 4.44  # ~225 mhz

HARDWARE_LIMITS = {
    "LUT":  117120,
    "BRAM": 288,    # 18k brams
    "DSP":  1248,
}

# vivado/vitis paths, only required when running real synthesis
XILINX_VIVADO_PATH = "/tools/Xilinx/Vivado/2024.2"
VITIS_HLS_PATH     = "/tools/Xilinx/Vitis_HLS/2024.2"
XILINX_VITIS_PATH  = "/tools/Xilinx/Vitis/2024.2"
FINN_RTLLIB_PATH   = "/home/max/finn-plus/finn-rtllib"

# nas search
POPULATION_SIZE  = 20
GENERATIONS      = 60
ELITISM          = 2
MUTATION_RATE    = 0.3
PROXY_EPOCHS     = 10     # epochs per genome during search
PROXY_BATCHES    = 100    # batches per epoch during search
SEARCH_SUBSET    = 800    # train images for proxy training
VAL_SUBSET       = 200    # val images for proxy validation
SEARCH_BATCH     = 4

# full training
EPOCHS         = 20
BATCH_SIZE     = 12
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-4
TRAIN_SPLIT    = 0.8
NUM_WORKERS    = 8

# model
NUM_ANGLES = 6  # k anchors for theta classification
