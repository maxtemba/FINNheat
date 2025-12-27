# FINN-Heat: Hardware-Aware NAS for Real-Time Grasping

FINN-Heat is a Neural Architecture Search (NAS) pipeline designed to discover efficient, quantized grasping models for Xilinx FPGAs (specifically the Kria KV260).

Unlike standard approaches that compress a large model (like ResNet-50) after training, this project searches for the optimal architecture from scratch. It evolves a "genome" of layers, channels, and bit-widths (4-bit/8-bit) to maximize accuracy while minimizing hardware latency.

## Project Structure

The project is organized into a core library, the main search engine, and testing scripts.

```text
.
├── nas_search.py           # The Evolutionary Search Engine. Runs the genetic algorithm.
├── dataset.py              # GraspNet data loader with on-the-fly target generation.
├── heatmap_generator.py    # Logic for Gaussian peaks and anchor encoding (HGGD).
├── core/
│   ├── models.py           # The Search Space. Defines the Dynamic NAS Model.
│   ├── hardware.py         # The Hardware Evaluator. Compiles to FINN HLS to estimate FPS/LUTs.
│   └── training.py         # The Accuracy Evaluator. Unified training loop and Loss function.
├── tests/
│   ├── train_genome.py     # Fully trains a discovered architecture (from best_genome.txt).
│   ├── verify_pred.py      # Runs inference on a single image to visualize predictions.
│   ├── export.py           # Exports the trained model to QONNX for the final FPGA build.
│   └── verify_gt.py        # Debug tool to check ground truth generation.
└── data/                   # Symlink or directory containing GraspNet-1Billion data.

```

## Prerequisites

* **Python 3.10+**
* **Brevitas:** For quantization-aware training.
* **FINN-Base:** For the hardware build flow and ONNX export.
* **Vivado/Vitis HLS:** Required only if running the hardware estimator locally.

## The Workflow

This pipeline follows a 4-step process: **Search → Train → Verify → Export**.

### 1. Run the Search (NAS)

Run the evolutionary algorithm to find the best architecture for your hardware constraints. This script spawns a population of random models, estimates their hardware performance (FPS) and accuracy (Loss), and evolves them over generations.

```bash
python nas_search.py

```

**Output:** Generates `best_nas_genome.txt`.

### 2. Full Training

Once the search identifies the best genome, train it fully on the dataset. This script reads the genome file to reconstruct the exact model architecture before training.

```bash
# Update GENOME_FILE path in the script if necessary
python tests/train_genome.py

```

**Output:** Saves weights to `trained_model_hggd.pth`.

### 3. Verify Predictions

Visual validation is critical for grasping. Run this script to generate a visualization (Input RGB vs. Heatmap Predictions) for a random test image.

```bash
python tests/verify_pred.py

```

**Output:** Saves `prediction_nas_3000.png`.

### 4. Export for FPGA

To deploy the model, export it to the QONNX intermediate representation. This format preserves the quantization metadata required by the FINN compiler.

```bash
python tests/export.py

```

**Output:** Generates `nas_model_export.onnx`.

## Core Components

### The Search Space (`core/models.py`)

Defines a `NAS_GHM_Model` constructed from `DynamicBlock` layers. The architecture is flexible, defined by a dictionary ("genome") that specifies:

* **Encoders:** Number of stages, kernel sizes (3x3 vs 5x5), channel widths, and bit-widths per stage.
* **Decoder:** Channel widths for upsampling.
    * **Quantization:** Mixed-precision support (4-bit or 8-bit weights).

### Hardware Estimator (`core/hardware.py`)

A "Turbo-Mode" estimator that acts as a proxy for the full FINN compiler. For every candidate model in the search, it:

1. Exports the model to ONNX.
2. Runs FINN cleanup transformations (Tidy, Streamline).
3. Calculates optimal folding factors (SIMD/PE) to saturate the FPGA.
4. Reports estimated FPS, Latency, and Resource Usage (LUT/BRAM).

### Heatmap Generator (`heatmap_generator.py`)

Implements the specific logic from the HGGD (Heatmap-based Grasp Generation) paper. It converts raw 5-DoF grasp rectangles into 5 dense tensors:

* **Location Map:** Gaussian confidence peaks.
* **Class Map:** Anchor bin classification.
* **Regression Maps:** Offsets for Angle, Width, and Depth.