import torch
import torch.nn as nn
import brevitas.nn as qnn
# ADDED: Int8ActPerTensorFloat for the input quantization
from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat, Int8ActPerTensorFloat

class QuantConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, stride=1):
        super().__init__()
        self.conv = qnn.QuantConv2d(
            in_channels, out_channels,
            kernel_size=kernel, padding=padding, stride=stride,
            weight_quant=Int8WeightPerTensorFloat,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            return_quant_tensor=False # Keep as False for intermediate blocks if you prefer,
            # but strictly typically internal layers pass QuantTensors.
            # For this specific fix, the Input is the priority.
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FINNCompatibleGHM_MultiOutput(nn.Module):
    def __init__(self, in_channels=4, num_angles=6, num_reg=2):
        super().__init__()

        # ======================================================================
        # 1. THE FIX: Explicit Input Quantization
        # ======================================================================
        # This node tells FINN: "Accept Float32 from CPU, convert to Int8,
        # and stream it into the FPGA."
        self.quant_input = qnn.QuantIdentity(
            bit_width=8,
            return_quant_tensor=True,  # Must be True to pass scale info to next layer
            act_quant=Int8ActPerTensorFloat # Signed Int8 allows for normalized inputs (negatives)
        )
        # ======================================================================

        # Encoder: conv-only
        # Stage 1: H/2, W/2
        self.enc1 = nn.Sequential(
            QuantConvBlock(in_channels, 32, stride=2),
            QuantConvBlock(32, 32)
        )

        # Stage 2: H/4, W/4
        self.enc2 = nn.Sequential(
            QuantConvBlock(32, 64, stride=2),
            QuantConvBlock(64, 64)
        )

        # Stage 3: H/8, W/8
        self.enc3 = nn.Sequential(
            QuantConvBlock(64, 96, stride=2),
            QuantConvBlock(96, 96)
        )

        # Bottleneck + small decoder
        self.bottleneck = nn.Sequential(
            QuantConvBlock(96, 128),
            QuantConvBlock(128, 96)
        )
        self.decoder = nn.Sequential(
            QuantConvBlock(96, 64),
            QuantConvBlock(64, 32)
        )

        # Heads
        self.head_confidence = qnn.QuantConv2d(
            32, 1, kernel_size=1,
            weight_quant=Int8WeightPerTensorFloat,
            bias=False
        )
        self.head_theta = qnn.QuantConv2d(
            32, num_angles, kernel_size=1,
            weight_quant=Int8WeightPerTensorFloat,
            bias=False
        )
        self.head_regression = qnn.QuantConv2d(
            32, num_reg, kernel_size=1,
            weight_quant=Int8WeightPerTensorFloat,
            bias=False
        )

    def forward(self, x):
        # ======================================================================
        # 2. THE FIX: Apply Input Quantization First
        # ======================================================================
        x = self.quant_input(x)
        # ======================================================================

        x = self.enc1(x)   # -> H/2
        x = self.enc2(x)   # -> H/4
        x = self.enc3(x)   # -> H/8
        feat = self.decoder(self.bottleneck(x))

        q_c_grid = self.head_confidence(feat)
        q_theta = self.head_theta(feat)
        q_reg = self.head_regression(feat)

        return q_c_grid, q_theta, q_reg