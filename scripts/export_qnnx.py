import os

from core.export import export_to_qonnx
from core.models import load_nas_model

# --- config paths
GENOME_FILE = "../genomes/best_genome.txt"
WEIGHTS_FILE = "outputs/trained_model.pth"
OUTPUT_ONNX = "outputs/model_export.onnx"

def main():
    print("starting model export...")

    # 1. build model and load weights
    try:
        model = load_nas_model(GENOME_FILE, weights_path=WEIGHTS_FILE, device='cpu')
        print("model built and weights loaded successfully.")
    except Exception as e:
        print(f"setup failed: {e}")
        print("train genome first!")
        return

    # 2. export
    os.makedirs("outputs", exist_ok=True)
    success = export_to_qonnx(model, OUTPUT_ONNX)

    if success:
        print(f"model saved to: {os.path.abspath(OUTPUT_ONNX)}")

if __name__ == "__main__":
    main()