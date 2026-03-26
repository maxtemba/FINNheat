# FINN-Heat: Hardware-Aware NAS for Real-Time Grasping

FINN-Heat is a Neural Architecture Search (NAS) pipeline that evolves quantized grasping models for Xilinx FPGAs (Kria KV260). It co-optimizes model accuracy and FPGA resource utilization from scratch using a genetic algorithm over quantized (4-bit/8-bit) architectures and hardware parallelism levels.

## Setup

**Requirements:** Ubuntu 22.04, Python 3.10, Xilinx Vivado/Vitis HLS 2024.2

### 1. Install Miniforge (conda)

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

### 2. Create the Python environment

```bash
conda create -n finn-plus python=3.10 -y
conda activate finn-plus
```

### 3. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install brevitas==0.10.2
pip install finn-plus
finn deps update
pip install numpy scikit-learn joblib matplotlib Pillow
```

### 4. Clone and install this project

```bash
git clone <your-repo-url>
cd FINNheat
pip install -e .
```

The `-e` install makes `core.*` importable from any script without path hacks.

### 5. Install Vivado/Vitis HLS 2024.2

Download the Xilinx Unified Installer from [xilinx.com](https://www.xilinx.com/support/download.html) and install to `/tools/Xilinx/`. You need at minimum:
- Vivado 2024.2
- Vitis HLS 2024.2

### 6. Set environment variables

Add to your `~/.bashrc` (or activate script for the conda env):

```bash
export XILINX_VIVADO=/tools/Xilinx/Vivado/2024.2
export VITIS_HLS=/tools/Xilinx/Vitis_HLS/2024.2
export XILINX_VITIS=/tools/Xilinx/Vitis/2024.2
export FINN_RTLLIB=/path/to/finn-rtllib   # from the finn package install
```

### 7. GraspNet dataset

Download the dataset and extract it into `data/graspnet/`:

```bash
# download
wget -O data.zip "https://YOUR_GOOGLE_DRIVE_LINK_HERE"
unzip data.zip -d data/
```

Expected layout:

```
FINNheat/
└── data/
    └── graspnet/
        ├── scenes/
        └── dataset_kinect/
```

## Project Structure

```
.
├── nas_search.py               # genetic algorithm search engine
├── genomes/
│   └── best_genome.txt         # current best genome (input to all scripts)
├── core/
│   ├── models.py               # NAS_GHM_Model + load_nas_model()
│   ├── evolution.py            # genome encoding, mutate, crossover, fitness
│   ├── export.py               # export_to_qonnx()
│   ├── hardware.py             # FINN+ estimator (FPS, LUTs, BRAMs, DSPs)
│   ├── training.py             # train_model() + hggd_loss()
│   ├── predictor.py            # calibrated RF predictor (genome → real LUT/BRAM/DSP)
│   ├── dataset.py              # GraspNetHeatmapDataset
│   └── heatmap_generator.py    # Gaussian heatmaps + anchor encoding
└── scripts/
    ├── train_genome.py               # full training of best_genome.txt
    ├── verify_pred.py                # visualize model predictions
    ├── verify_gt.py                  # visualize ground truth heatmaps
    ├── export_qnnx.py                # export trained model to QONNX
    ├── predict_hardware_genome.py    # run hardware estimator
    ├── synthesize_hardware_genome.py # full HLS + OOC synthesis
    ├── collect_calibration_data.py   # synthesize genomes to collect predictor training data
    ├── train_predictor.py            # train calibrated hw predictor from synthesis data
    └── _synth_worker.py              # subprocess worker for synthesis (called by collect script)
```

## Workflow

### 1. (Optional) Calibrate Hardware Predictor

The NAS uses calibrated RF predictors to correct FINN's LUT/BRAM/DSP estimates. Predictors are trained per parallelism level (2, 3, 4). Pre-trained predictors are stored in `scripts/outputs/predictors/`.

To collect new synthesis data or retrain a predictor:

```bash
# set PARALLELISM in collect_calibration_data.py, then:
cd scripts && python collect_calibration_data.py   # ~30–120 min per genome

# set PARALLELISM in train_predictor.py to match, then:
cd scripts && python train_predictor.py
```

Output: `scripts/outputs/predictors/hw_predictor_p{N}.pkl`

### 2. NAS Search

Runs the genetic algorithm (population 20, 90 generations). Each candidate genome encodes architecture **and** parallelism level (2×, 3×, or 4× PE/SIMD folding). Candidates are hardware-estimated via FINN+, calibrated by the matching predictor, and proxy-trained before scoring.

```bash
python nas_search.py
```

Outputs `genomes/best_nas_genome.txt` and a per-individual log at `nas_search_log.csv`. Copy the best genome to `genomes/best_genome.txt` to use it downstream.

### 3. Full Training

```bash
cd scripts && python train_genome.py
```

Outputs `scripts/trained_model.pth`.

### 4. Verify

```bash
cd scripts && python verify_pred.py   # predictions vs input
cd scripts && python verify_gt.py     # ground truth heatmaps
```

### 5. Export for FPGA

```bash
cd scripts && python export_qnnx.py
```

Outputs `scripts/model_export.onnx` with INT4/INT8 quantization metadata preserved for FINN+.

### 6. Hardware Estimation / Synthesis

```bash
cd scripts && python predict_hardware_genome.py       # fast estimate
cd scripts && python synthesize_hardware_genome.py    # full HLS + OOC (~30–120 min)
```

## Genome Encoding

Each architecture is a dict with 7 parameters:

| Gene | Values | Description |
|------|--------|-------------|
| `enc_ch` | [16–128] × 3 | encoder channels per stage, monotonically increasing |
| `enc_depth` | [1–3] × 3 | encoder block depth per stage |
| `enc_k` | [3, 5] × 3 | encoder kernel sizes per stage |
| `enc_bits` | [4, 8] × 3 | quantization bit-widths per stage |
| `btl_ch` | 64–256 | bottleneck channels |
| `dec_ch` | [16–96] × 3 | decoder channels per stage, monotonically decreasing |
| `parallelism` | 2, 3, 4 | PE/SIMD folding target for all MVAU layers |

`parallelism` controls the FINN hardware folding: higher values mean more FPS but more LUTs. Each level has its own calibrated predictor (`hw_predictor_p{N}.pkl`).

## Hardware Target (Kria KV260)

| Resource | Limit |
|----------|-------|
| LUTs     | 117,120 |
| BRAMs    | 288 (18K) |
| DSPs     | 1,248 |
| Clock    | 300 MHz |
