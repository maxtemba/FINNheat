import os
import ast
import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat, Int8ActPerTensorFloat


class QuantBlock(nn.Module):
    # quantized conv -> batchnorm -> quant relu
    def __init__(self, in_c, out_c, k, s, bits):  # in ch, out ch, kernel, stride, bitwidth
        super().__init__()
        pad = (k - 1) // 2  # keep spatial size constant
        self.block = nn.Sequential(
            qnn.QuantConv2d(
                in_c, out_c, kernel_size=k, stride=s, padding=pad,
                weight_bit_width=bits, bias=False,
                weight_quant=Int8WeightPerTensorFloat
            ),
            nn.BatchNorm2d(out_c),
            qnn.QuantReLU(bit_width=bits, act_quant=Uint8ActPerTensorFloat)
        )

    def forward(self, x):
        return self.block(x)


class NAS_GHM_Model(nn.Module):
    def __init__(self, genome, num_angles=6):
        super().__init__()

        # input quantized to 8 bit, finn requirement
        self.quant_input = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True, act_quant=Int8ActPerTensorFloat)

        self.stages = nn.ModuleList()
        in_c = 4

        # encoder from genome
        for i in range(3):
            out_c = genome['enc_ch'][i]
            depth = genome['enc_depth'][i]
            k     = genome['enc_k'][i]
            bits  = genome['enc_bits'][i]

            layers = [QuantBlock(in_c, out_c, k, 2, bits)]  # downsample
            for _ in range(depth - 1):
                layers.append(QuantBlock(out_c, out_c, k, 1, bits))  # process

            self.stages.append(nn.Sequential(*layers))
            in_c = out_c

        # bottleneck
        btl_c = genome['btl_ch']
        self.bottleneck = nn.Sequential(
            QuantBlock(in_c, btl_c, 1, 1, 8),
            QuantBlock(btl_c, in_c, 1, 1, 8)
        )

        # decoder
        self.dec_stages = nn.ModuleList()
        for out_c in genome['dec_ch']:
            self.dec_stages.append(QuantBlock(in_c, out_c, 1, 1, 8))
            in_c = out_c

        # output heads
        def make_head(out_ch):
            return qnn.QuantConv2d(in_c, out_ch, 1, weight_bit_width=8, bias=True, weight_quant=Int8WeightPerTensorFloat)

        self.head_loc = make_head(1)
        self.head_cls = make_head(num_angles)
        self.head_theta = make_head(num_angles)
        self.head_width = make_head(num_angles)
        self.head_depth = make_head(num_angles)

        # init loc/cls bias to -log((1-0.01)/0.01) ~ -4.59 so initial preds are ~0.01
        # prevents focal loss from wasting epochs pushing background to zero
        for head in (self.head_loc, self.head_cls):
            if head.bias is not None:
                torch.nn.init.constant_(head.bias, -4.59)

    def forward(self, x):
        x = self.quant_input(x)

        for stage in self.stages:
            x = stage(x)

        x = self.bottleneck(x)

        for stage in self.dec_stages:
            x = stage(x)

        return (self.head_loc(x), self.head_cls(x), self.head_theta(x), self.head_width(x), self.head_depth(x))


def load_nas_model(genome_path, weights_path=None, device='cpu'):
    # load genome from file, build model, optionally load weights
    if not os.path.exists(genome_path):
        raise FileNotFoundError(f"genome file not found at {genome_path}")

    with open(genome_path, "r") as f:
        genome = ast.literal_eval(f.read().strip())

    print(f"loaded genome: {genome}")

    try:
        model = NAS_GHM_Model(genome).to(device)
    except Exception as e:
        raise RuntimeError(f"failed to build model from genome: {e}")

    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"weights file not found at {weights_path}")

        print(f"loading weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

    return model
