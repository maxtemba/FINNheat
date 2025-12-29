import sys
import os

sys.path.append("..")
from core.hardware import export_to_qonnx
from core.utils import load_nas_model

# --- config paths
GENOME_FILE = "best_genome.txt"
WEIGHTS_FILE = "trained_model_hggd.pth"
OUTPUT_ONNX = "nas_model_export.onnx"

def main():
    print("starting model export...")

    # 1. build model and load weights uses logic from core/utils.py
    try:
        model = load_nas_model(GENOME_FILE, weights_path=WEIGHTS_FILE, device='cpu')
        print("model built and weights loaded successfully.")
    except Exception as e:
        print(f"setup failed: {e}")
        print("train genome first!")
        return

    # 2. export using core/hardware.py logic
    success = export_to_qonnx(model, OUTPUT_ONNX)

    if success:
        print(f"model saved to: {os.path.abspath(OUTPUT_ONNX)}")

if __name__ == "__main__":
    main()