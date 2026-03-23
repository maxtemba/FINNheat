import torch
import brevitas.onnx as bo


def export_to_qonnx(model, filename):
    """
    exports pytorch model as quantized ONNX file uses dummy input for it.

    :param model: pytorch model.
    :param filename: path to export ONNX file.
    :return: true/false.
    """
    model.eval()
    original_device = next(model.parameters()).device
    model.cpu()
    dummy = torch.randn(1, 4, 360, 640)
    print(f"exporting QONNX model to: {filename}")
    try:
        bo.export_qonnx(model, input_t=dummy, export_path=filename)
        return True
    except Exception as e:
        print(f"export failed: {e}")
        return False
    finally:
        model.to(original_device)
