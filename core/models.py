import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat, Int8ActPerTensorFloat

class DynamicBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s, bits):
        super().__init__()
        pad = (k - 1) // 2
        # Standard Quantized Conv (Most stable for FINN)
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
        self.quant_input = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True, act_quant=Int8ActPerTensorFloat)

        self.stages = nn.ModuleList()
        in_c = 4

        # --- Encoder (Gene-driven) ---
        for i in range(3):
            out_c = genome['enc_ch'][i]
            depth = genome['enc_depth'][i]
            k     = genome['enc_k'][i]
            bits  = genome['enc_bits'][i]

            layers = [DynamicBlock(in_c, out_c, k, 2, bits)] # Downsample
            for _ in range(depth - 1):
                layers.append(DynamicBlock(out_c, out_c, k, 1, bits)) # Process

            self.stages.append(nn.Sequential(*layers))
            in_c = out_c

        # --- Bottleneck ---
        btl_c = genome['btl_ch']
        self.bottleneck = nn.Sequential(
            DynamicBlock(in_c, btl_c, 3, 1, 8),
            DynamicBlock(btl_c, in_c, 3, 1, 8)
        )

        # --- Decoder ---
        dec_layers = []
        for out_c in genome['dec_ch']:
            dec_layers.append(DynamicBlock(in_c, out_c, 3, 1, 8))
            in_c = out_c
        self.decoder = nn.Sequential(*dec_layers)

        # --- Heads ---
        def make_head(out_ch):
            return qnn.QuantConv2d(in_c, out_ch, 1, weight_bit_width=8, bias=True, weight_quant=Int8WeightPerTensorFloat)

        self.head_loc = make_head(1)
        self.head_cls = make_head(num_angles)
        self.head_theta = make_head(num_angles)
        self.head_width = make_head(num_angles)
        self.head_depth = make_head(num_angles)

    def forward(self, x):
        x = self.quant_input(x)
        for stage in self.stages: x = stage(x)
        feat = self.decoder(self.bottleneck(x))
        return (self.head_loc(feat), self.head_cls(feat), self.head_theta(feat), self.head_width(feat), self.head_depth(feat))