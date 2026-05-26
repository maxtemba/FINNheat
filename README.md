# FINN-Heat

Hardware-aware NAS pipeline that evolves quantized grasping models for Xilinx FPGAs (Kria KV260). Co-optimizes model accuracy and FPGA resource utilization using a genetic algorithm over 4-bit/8-bit quantized architectures.

## Setup

Requires Python 3.10. Hardware synthesis additionally needs Ubuntu 22.04 and Xilinx Vivado/Vitis HLS 2024.2 installed at `/tools/Xilinx/`.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install brevitas==0.10.2 finn-plus numpy scikit-learn joblib matplotlib Pillow
finn deps update
```

Extract the GraspNet dataset into `data/graspnet/`:

```
FINNheat/data/graspnet/
├── scenes/
└── dataset_kinect/
```

## Usage

```bash
# NAS search — writes genomes/best_nas_genome.txt + results/nas_search_log.csv
python nas_search.py

# Full training — trains scripts/outputs/gen33_genome.txt -> scripts/outputs/gen33_full_model.pth
cd scripts/training && python train_genome.py

# Export trained model to QONNX for FINN+
cd scripts/hardware && python export_qnnx.py

# Hardware estimate / full synthesis
cd scripts/hardware && python predict_hardware_genome.py       # fast RF estimate
cd scripts/hardware && python synthesize_hardware_genome.py    # full HLS + OOC (~30–120 min)

# (Optional) re-calibrate the hardware predictor; pre-trained pkls live in scripts/outputs/predictors/
cd scripts/predictor && python collect_calibration_data.py    # set PARALLELISM first
cd scripts/predictor && python train_predictor.py
```

## Project Structure

```
config.py                              central paths + hyperparameters
nas_search.py                          genetic algorithm search
core/                                  pipeline modules (models, training, hardware, dataset, ...)
scripts/
  training/      train_genome.py, build_image_cache.py
  hardware/      export_qnnx.py, predict_hardware_genome.py, synthesize_hardware_genome.py
  predictor/     collect_calibration_data.py, train_predictor.py, _synth_worker.py
  outputs/       active genome, trained weights, predictor pkls, calibration JSON
genomes/                               NAS-saved best genomes
results/                               experiment CSVs and run logs
```

## Genome Encoding

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
