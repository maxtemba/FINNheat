# FINN-Compatible Grasp Heatmap Model

This project contains a complete, end-to-end pipeline for training and analyzing a FINN-compatible grasp heatmap detection model. It is designed to work with the GraspNet dataset and provides a full workflow from raw data to a hardware performance analysis for FPGA deployment.

This repository is a proof-of-concept demonstrating how to:

1.  Handle the complex GraspNet dataset structure.
2.  Generate grasp heatmaps on-the-fly, as described in the HGGD paper.
3.  Design a quantized, linear (no skip-connection) FCN model that is compatible with the FINN builder.
4.  Train the model on the 25,600-image dataset.
5.  Visually evaluate the trained model's predictions.
6.  Analyze the final model's hardware performance for real-time FPGA deployment.

-----

## 🚀 Key Features

  * **FINN-Compatible Model:** A custom, linear FCN (`FINNCompatibleGHM_MultiOutput`) built with Brevitas for 8-bit quantization. It features three separate output heads to predict confidence, angle, and grasp attributes.
  * **Advanced Data Pipeline:** `dataset.py` and `heatmap_generator.py` successfully navigate the complex, mismatched GraspNet dataset structure, parse raw `.npz` label files, and generate ground-truth heatmaps on-the-fly.
  * **End-to-End Workflow:** Includes scripts to train the model (`train.py`), visually evaluate its predictions (`evaluate.py`), and analyze its hardware performance (`check_model.py`).

-----

## 📈 Performance Results

The final trained model, when analyzed by the FINN builder (targeting a Pynq-Z1), achieves an estimated **32.15 FPS throughput**, proving its suitability for real-time applications.

-----

## 🗺️ How to Replicate This Project

Here is the step-by-step recipe to replicate this project.

### Step 1: Environment Setup

#### 1\. Get the Data

  * **GraspNet Images:** Download the GraspNet "Train Images" (parts 1-4). Unzip them all into a single `graspnet/scenes/` folder.
  * **GraspNet Labels:** Download the authors' preprocessed labels (`Kinect Dataset.7z` and/or `RealSense Dataset.7z`). Unzip them into their respective folders (e.g., `graspnet/dataset_kinect/`).
  * **Verify Structure:** Your final data structure should look like this, with mismatched scene names and nested folders:
    ```
    graspnet/
    ├── scenes/
    │   └── scene_0000/
    │       └── kinect/
    │           ├── rgb/
    │           │   └── 0000.png
    │           └── depth/
    │               └── 0000.png
    └── dataset_kinect/
        └── scene_0/
            └── grasp_labels/
                └── 0_view.npz
    ```

#### 2\. Create Python Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

#### 3\. Install All Libraries

```bash
pip install torch torchvision
pip install brevitas onnx onnxruntime qonnx
pip install finn-plus
pip install matplotlib numpy
```

#### 4\. (For macOS/Linux Users)

If you are running analysis on a machine without the Xilinx tools installed, set this dummy environment variable to allow the reporting steps to run:

```bash
export XILINX_VIVADO="/not/installed"
```

-----

### Step 2: Create Your Project Files

Create the following six Python files in your project directory.

  * **`model.py`**

      * Contains the `QuantConvBlock` class for a quantized Conv-BN-ReLU layer.
      * Contains the `FINNCompatibleGHM_MultiOutput` class.
      * **Crucial Logic:** The model is a linear FCN (no skip-connections) to make it compatible with the automated FINN builder. It has three separate output heads: `head_confidence`, `head_theta`, and `head_regression`.

  * **`heatmap_generator.py`**

      * Contains the `HeatmapGenerator` class.
      * **Crucial Logic:** The `generate_ground_truth` function implements the "Gaussian encoding" (for confidence) and "grid-based strategy" (for attributes) from the original paper.

  * **`dataset.py`**

      * Contains the `GraspNetHeatmapDataset` class.
      * **Crucial Logic (File Paths):** Correctly navigates the mismatched paths (e.g., `scenes/scene_0000/kinect/rgb/0000.png` and `dataset_kinect/scene_0/grasp_labels/0_view.npz`).
      * **Crucial Logic (Data Keys):** Loads the raw `.npz` file and builds the `(N, 5)` grasp array by stacking the `centers_2d`, `thetas_rad`, `widths_2d`, and `center_z_depths` arrays.
      * **Crucial Logic (Outputs):** Returns three separate ground-truth tensors: `y_conf_low_res`, `y_theta_low_res`, and `y_reg_low_res`.

  * **`train.py`**

      * Loads the model and dataset for training.
      * **Crucial Logic:** Implements the `combined_loss` function, which correctly calculates three separate losses (BCE for confidence, CrossEntropy for theta, and L1 for regression) for the three heads and adds them together.
      * Saves the final `trained_model.pth`.

  * **`evaluate.py`**

      * Loads the `trained_model.pth` and a single item from `GraspNetHeatmapDataset`.
      * **Crucial Logic:** Re-loads the raw `.npz` data using the correct keys (just like `dataset.py`) to generate the full-resolution ground-truth for comparison.
      * Saves the `prediction_dashboard.png` for visual checking.

  * **`check_model.py`**

      * Loads the final `trained_model.pth`.
      * **Crucial Logic (Export):** Exports the trained model to `trained_heatmap_model.onnx` using `bo.export_qonnx`.
      * **Crucial Logic (Analysis):** Configures `DataflowBuildConfig` with `target_fps = 100` to enable optimization (folding) in the FINN builder.
      * **Crucial Logic (Error Handling):** Uses a `try...except` block to catch the expected `KeyError: 'FINN_RTLLIB'` on non-Xilinx systems, allowing the script to proceed and print the generated report.

-----

### Step 3: Run the Workflow

1.  **Train the Model:**

    ```bash
    python train.py
    ```

    Wait for the training to complete and for `trained_model.pth` to be created.

2.  **Visually Check the Model:**

    ```bash
    python evaluate.py
    ```

    Open the generated `prediction_dashboard.png` to confirm the model learned to predict grasp locations.

3.  **Get FINN Performance:**

    ```bash
    python check_model.py
    ```

    Read the "ANALYTICAL PREDICTIONS" table from the console output to see the estimated FPS and latency.
