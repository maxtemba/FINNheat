### `README.md`

```markdown
# 🦾 FINN-Heat: Real-Time Grasping on FPGA (Baseline)

> **A proof-of-concept for running a linear, 8-bit quantized grasp detection model on Xilinx FPGAs.**

---

![Ground Truth vs Prediction](https://github.com/user-attachments/assets/placeholder-for-your-image-link)
*(Left: Ground Truth Target | Right: Actual Model Prediction from this repo)*

## 🤔 The Problem
Robotic grasping usually relies on massive, heavy neural networks (like ResNet-50 or U-Net) that require power-hungry GPUs.

But if you want to run this on the **Edge**—specifically on a Xilinx FPGA (like the Kria KV260 or Pynq-Z1)—you hit a wall:
* **Skip Connections** (ResNet/U-Net) are difficult to implement in hardware streams.
* **Float32 Math** kills throughput and consumes too much logic.
* **Standard Architectures** are often too large for on-chip memory (BRAM).

## 💡 The Solution
**FINN-Heat** is a custom pipeline designed to prove that complex grasping tasks *can* run on simple hardware.

We built a **Linear, Quantized Fully-Convolutional Network (FCN)** that serves as a hardware-friendly baseline:
1.  **Stream-Friendly:** No skip connections. Data flows in one direction (Input → Output).
2.  **8-bit Quantized:** Built with `Brevitas` to train with integers from day one.
3.  **Dense Prediction:** Outputs high-resolution heatmaps ($45 \times 80$) for accurate grasp positioning, matching the **HGGD Paper's** methodology.

> **Note:** This repository currently contains a single, manually designed model (`FINNCompatibleGHM_MultiOutput`). It serves as the valid **baseline** for future Neural Architecture Search (NAS) experiments.

## ⚡ Key Features

* **GHM Architecture:** A custom 5-head model that predicts **Confidence**, **Angle**, **Width**, and **Depth** simultaneously.
* **FINN-Ready:** Designed specifically for the Xilinx FINN compiler (Linear topology, Int8 weights/activations).
* **On-the-Fly Generation:** The `HeatmapGenerator` converts raw GraspNet labels into training targets in real-time—no massive pre-processing needed.
* **Hardware Analyzer:** Includes `check_model.py` to estimate real FPS and resource usage on a KV260 or Pynq-Z1 board.

---

## 📸 Visual Verification

Does this simplified linear model actually learn? **Yes.**
Below is a comparison from our validation script. The model successfully learns to ignore the table (background) and predicts specific grasp angles (colors) for the objects.

| Ground Truth (Target) | Model Prediction (Raw Output) |
| :---: | :---: |
| <img src="gt_raw_3000.png" width="400"> | <img src="prediction_raw_3000.png" width="400"> |
| *Clean Gaussian peaks derived from annotations* | *Dense predictions from our linear quantized model* |

---

## 🛠️ How to Run It

### 1. Setup Environment
You need a machine with PyTorch and Brevitas installed.
```bash
# Create env
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision brevitas finn-plus matplotlib numpy onnx

```

### 2. Prepare Data

Download the **GraspNet-1Billion** subset (or full set) and arrange it like this:

```
data/graspnet/
├── scenes/             # RGB-D Images
└── dataset_kinect/     # Raw Labels (.npz)

```

### 3. Train the Baseline

This trains the hard-coded linear model using the HGGD Loss function.

```bash
python train_cpu.py

```

*Target Loss:* You should see the loss drop from `~60.0` to `~0.6` within 10-15 epochs.

### 4. Verify Predictions

Visually check that the model is learning correctly (saves `prediction_raw_XXXX.png`).

```bash
python tests/verify_pred.py

```

### 5. Hardware Analysis

Curious how fast this runs on a chip? Run the hardware checker.

```bash
python check_model.py

```

**Expected Output:**

> 🚀 FPS: ~32.15
> 🐢 Bottleneck: 200MHz

---

## 🔮 Future Work (NAS)

Currently, this repository uses a fixed architecture (`model.py`).
The next phase of this project is to implement **Neural Architecture Search (NAS)** to automatically find models that are even faster or smaller than this baseline, optimizing specifically for the number of LUTs and DSPs available on the target FPGA.

## 📝 Credits & References

* **Original Concept:** "HGGD: A Heatmap-based Grasp Generation Method"
* **Compiler:** Xilinx [FINN](https://github.com/Xilinx/finn) framework.
* **Quantization:** AMD/Xilinx [Brevitas](https://github.com/Xilinx/brevitas).

---

*Created by Max Temba as a Proof-of-Concept for efficient edge robotics.*

```

```