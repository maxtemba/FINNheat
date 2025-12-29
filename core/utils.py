import os
import ast
import torch
import sys

from core.models import NAS_GHM_Model

def load_nas_model(genome_path, weights_path=None, device='cpu'):
    """
    creates a model from a genome with optional with weight.
    :return pytorch model.
    """

    # 1. load genome
    if not os.path.exists(genome_path):
        raise FileNotFoundError(f"genome file not found at {genome_path}")

    with open(genome_path, "r") as f:
        genome = ast.literal_eval(f.read().strip())

    print(f"loaded genome: {genome}")

    # 2. build model
    try:
        model = NAS_GHM_Model(genome).to(device)
    except Exception as e:
        raise RuntimeError(f"failed to build model from genome: {e}")

    # 3. load weights (optional)
    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"weights file not found at {weights_path}")

        print(f"loading weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

    return model