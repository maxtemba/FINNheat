# FINN-Heat: Hardware-Aware NAS for Real-Time Grasping

FINN-Heat is a Neural Architecture Search (NAS) pipeline that evolves quantized grasping models for Xilinx FPGAs (Kria KV260). It co-optimizes model accuracy and FPGA resource utilization from scratch using a genetic algorithm over quantized (4-bit/8-bit) architectures.

## Setup

This project requires the `finn-plus` conda environment. Do **not** use pip or a venv.

```bash
conda activate finn-plus
```

Xilinx Vivado/Vitis 2024.2 must be installed at `/tools/Xilinx/`.
GraspNet dataset goes in `data/graspnet/` relative to the project root.

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
│   ├── dataset.py              # GraspNetHeatmapDataset
│   └── heatmap_generator.py    # Gaussian heatmaps + anchor encoding
└── scripts/
    ├── train_genome.py         # full training of best_genome.txt
    ├── verify_pred.py          # visualize model predictions
    ├── verify_gt.py            # visualize ground truth heatmaps
    ├── export_qnnx.py          # export trained model to QONNX
    ├── predict_hardware_genome.py    # run hardware estimator
    └── synthesize_hardware_genome.py # full HLS + OOC synthesis
```

## Workflow

### 1. NAS Search

Runs the genetic algorithm (population 20, 400 generations). Each candidate is hardware-estimated via FINN+ and proxy-trained before scoring.

```bash
python nas_search.py
```

Outputs `genomes/best_nas_genome.txt`. Copy it to `genomes/best_genome.txt` to use it downstream.

### 2. Full Training

```bash
cd scripts && python train_genome.py
```

Outputs `scripts/trained_model.pth`.

### 3. Verify

```bash
cd scripts && python verify_pred.py   # predictions vs input
cd scripts && python verify_gt.py     # ground truth heatmaps
```

### 4. Export for FPGA

```bash
cd scripts && python export_qnnx.py
```

Outputs `scripts/model_export.onnx` with INT4/INT8 quantization metadata preserved for FINN+.

### 5. Hardware Estimation / Synthesis

```bash
cd scripts && python predict_hardware_genome.py       # fast estimate
cd scripts && python synthesize_hardware_genome.py    # full HLS + OOC (~30–120 min)
```

## Hardware Target (Kria KV260)

| Resource | Limit |
|----------|-------|
| LUTs     | 117,120 |
| BRAMs    | 288 (18K) |
| DSPs     | 1,248 |
| Clock    | 300 MHz |
