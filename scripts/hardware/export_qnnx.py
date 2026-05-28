import os

import config
from core.export import export_to_qonnx
from core.models import load_nas_model

OUTPUT_ONNX = os.path.join(config.OUTPUTS_DIR, "model_export.onnx")


def main():
    print("starting model export...")

    # build model and load weights
    try:
        model = load_nas_model(config.ACTIVE_GENOME, weights_path=config.ACTIVE_WEIGHTS, device='cpu')
        print("model built and weights loaded successfully.")
    except Exception as e:
        print(f"setup failed: {e}")
        print("train genome first!")
        return

    # export
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    success = export_to_qonnx(model, OUTPUT_ONNX)

    if success:
        print(f"model saved to: {os.path.abspath(OUTPUT_ONNX)}")


if __name__ == "__main__":
    main()
