# FINN-Heat

Hardware-aware NAS pipeline that evolves quantized grasping models for Xilinx FPGAs (Kria KV260). Co-optimizes model accuracy and FPGA resource utilization using a genetic algorithm over 4-bit/8-bit quantized architectures.

## Setup

**Requirements:** Ubuntu 22.04, Python 3.10, Xilinx Vivado/Vitis HLS 2024.2

### 1. Install Miniforge

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

### 2. Create environment

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

### 4. Install project

```bash
git clone <repo-url>
cd FINNheat
pip install -e .
```

### 5. Vivado/Vitis HLS 2024.2

Install from [xilinx.com](https://www.xilinx.com/support/download.html) to `/tools/Xilinx/`. Required for hardware synthesis only.

### 6. GraspNet dataset

Extract into `data/graspnet/`:

```
FINNheat/data/graspnet/
├── scenes/
└── dataset_kinect/
```

## Usage

Always activate the conda env first:
```bash
conda activate finn-plus
```

### NAS Search
```bash
python nas_search.py
```
Runs genetic algorithm (population 20, 90 generations). Outputs `scripts/outputs/best_nas_genome.txt` and `nas_search_log.csv`.

### Full Training
```bash
cd scripts && python train_genome.py
```
Trains the genome at `scripts/outputs/gen33_genome.txt`. Outputs `scripts/outputs/gen33_full_model.pth`.

### Verify
```bash
cd scripts && python verify_pred.py   # visualize predictions
cd scripts && python verify_gt.py     # visualize ground truth
```

### Export to FPGA
```bash
cd scripts && python export_qnnx.py
```
Exports trained model to QONNX with INT4/INT8 metadata for FINN+.

### Hardware Estimation
```bash
cd scripts && python predict_hardware_genome.py       # fast RF estimate
cd scripts && python synthesize_hardware_genome.py    # full HLS + OOC (~30–120 min)
```

### Calibrate Hardware Predictor (optional)
Pre-trained predictors are in `scripts/outputs/predictors/`. To retrain:
```bash
cd scripts && python collect_calibration_data.py   # synthesize genomes, set PARALLELISM first
cd scripts && python train_predictor.py            # train RF predictor from collected data
```

## Project Structure

```
nas_search.py                          genetic algorithm search
core/
  models.py                            NAS_GHM_Model, load_nas_model()
  evolution.py                         genome encoding, mutate, crossover, fitness
  training.py                          train_model(), hggd_loss()
  hardware.py                          FINN+ hardware estimator (FPS/LUTs/BRAMs/DSPs)
  predictor.py                         calibrated random forest predictor
  dataset.py                           GraspNetHeatmapDataset
  heatmap_generator.py                 Gaussian heatmaps + anchor encoding
  export.py                            export_to_qonnx()
scripts/
  train_genome.py                      full training of a genome
  verify_pred.py                       visualize model predictions
  verify_gt.py                         visualize ground truth heatmaps
  export_qnnx.py                       export to QONNX for FINN+
  predict_hardware_genome.py           run hardware estimator on a genome
  synthesize_hardware_genome.py        full HLS + out-of-context synthesis
  collect_calibration_data.py          synthesize genomes to build predictor training data
  train_predictor.py                   train calibrated hardware predictor
  build_image_cache.py                 pre-cache dataset images as .npy for faster loading
  _synth_worker.py                     subprocess worker called by collect script
  outputs/
    gen33_genome.txt                   active genome used for training
    gen33_full_model.pth               trained model weights
    predictors/hw_predictor_p{2,3,4}.pkl  calibrated RF predictors per parallelism level
    calibration_data_p{2,3,4}.json    raw synthesis data for predictor training
```

## Genome Encoding

Each genome is a dict with 7 parameters:

| Gene | Values | Description |
|------|--------|-------------|
| `enc_ch` | [16–128] × 3 | encoder channels per stage |
| `enc_depth` | [1–3] × 3 | blocks per encoder stage |
| `enc_k` | [3, 5] × 3 | kernel sizes per stage |
| `enc_bits` | [4, 8] × 3 | quantization bit-widths per stage |
| `btl_ch` | 64–256 | bottleneck channels |
| `dec_ch` | [16–96] × 3 | decoder channels per stage |
| `parallelism` | 2, 3, 4 | PE/SIMD folding level |

## Hardware Target (Kria KV260)

| Resource | Limit |
|----------|-------|
| LUTs | 117,120 |
| BRAMs | 288 (18K) |
| DSPs | 1,248 |
| Clock | 225 MHz |
