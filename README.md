FINN-Heat: Hardware-Aware NAS for Real-Time Grasping
FINN-Heat is a Neural Architecture Search (NAS) pipeline designed to discover efficient, quantized grasping models for Xilinx FPGAs (specifically the Kria KV260).

Unlike standard approaches that compress a huge model after training, this project searches for the optimal architecture from scratch. It evolves a "genome" of layers, channels, and bit-widths (4-bit/8-bit) to find the sweet spot between high accuracy and low latency.

Setup & Installation
This project uses FINN+ (by EKI Project), which allows for a clean, containerless installation without requiring Docker.

1. Environment Create a clean python environment (Python 3.10 required).

Bash

python -m venv venv
source venv/bin/activate
2. Install FINN+ and Dependencies Install the finn-plus package directly via pip. This pulls in Brevitas and the QONNX tools automatically.

Bash

# 1. Install standard helpers
pip install numpy pandas pillow tqdm matplotlib

# 2. Install FINN+ (Containerless Compiler)
pip install finn-plus

# 3. Pull external Vivado/HLS dependencies
finn deps update
3. Xilinx Tools Ensure you have Vivado/Vitis HLS (2022.2 or 2024.2) installed and in your PATH. FINN+ uses these locally to run the hardware estimation and synthesis.

Project Structure
The project is organized into the core logic, the search engine, and the testing suite.

Plaintext

.
├── nas_search.py           # The Search Engine. Runs the genetic algorithm.
├── core/
│   ├── dataset.py          # Clean GraspNet loader with on-the-fly target generation.
│   ├── evolution.py        # The Genetic Algorithm logic (mutate, crossover).
│   ├── hardware.py         # The Estimator. Compiles to FINN HLS to predict FPS/LUTs.
│   ├── heatmap_generator.py # Physics-based logic for Gaussian peaks and anchors.
│   ├── models.py           # The Search Space. Defines the Dynamic NAS Model.
│   ├── training.py         # The Trainer. Unified loop and HGGD Loss function.
│   └── utils.py            # Helpers to load/save genomes and weights.
├── tests/
│   ├── train_genome.py     # Fully trains a winner architecture (from best_genome.txt).
│   ├── verify_pred.py      # Runs inference on images to visualize predictions.
│   ├── export_qnnx.py      # Exports the trained model to QONNX for the FPGA build.
│   └── verify_gt.py        # Debug tool to check ground truth generation.
└── data/                   # Directory containing GraspNet data.
The Workflow
This pipeline follows a 4-step process: Search → Train → Verify → Export.

1. Run the Search (NAS)
   Start the evolutionary algorithm. It spawns a population of random models, estimates their hardware performance (FPS) on the Kria KV260 using the local FINN+ install, and evolves them over generations.

Bash

python nas_search.py
Output: Generates best_nas_genome.txt and logs history to nas_search_log.csv.

2. Full Training
   Once you have a winner, train it fully on the dataset. This script reconstructs the model from the genome file and runs a complete training cycle with detailed logging.

Bash

python tests/train_genome.py
Output: Saves weights to trained_model_hggd.pth and stats to training_log.csv.

3. Verify Predictions
   Visual validation is critical. Run this to see how your model performs on a random test image compared to the ground truth.

Bash

python tests/verify_pred.py
Output: Generates prediction_nas.png.

4. Export for FPGA
   To deploy, export the trained model to QONNX. This format preserves the quantization metadata (INT4/INT8) required by the FINN+ compiler.

Bash

python tests/export_qnnx.py
Output: Generates model.qonnx.

Core Mechanics
Hardware Estimator (core/hardware.py)
A custom proxy for the FINN+ compiler. Because we use finn-plus, we can invoke the build steps directly from Python without launching a Docker container. It:

Exports the candidate to QONNX.

Runs cleanup transformations (Tidy, Streamline).

Calculates optimal "Folding Factors" (Parallelism vs. Resources).

Reports estimated FPS (targeting 144Hz+) and LUT/BRAM usage.

Heatmap Generator (core/heatmap_generator.py)
Implements the HGGD paper logic. It converts raw 5-DoF grasp rectangles into 5 dense training tensors:

Location: Gaussian confidence peaks.

Class: Anchor bin classification.

Regression: Offsets for Angle, Width, and Depth.