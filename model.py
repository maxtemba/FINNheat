import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Uint8ActPerTensorFloat

class QuantConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, stride=1):
        super(QuantConvBlock, self).__init__()
        self.conv = qnn.QuantConv2d(
            in_channels, out_channels,
            kernel_size=kernel, padding=padding, stride=stride,
            weight_quant=Int8WeightPerTensorFloat,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            return_quant_tensor=False
        )
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FINNCompatibleGHM_MultiOutput(nn.Module):
    """
    FINN-compatible model with 3 separate output heads
    to predict confidence, theta (angle class), and regression (w, d).
    """
    def __init__(self, in_channels=4, num_angles=6, num_reg=2):
        super(FINNCompatibleGHM_MultiOutput, self).__init__()

        # --- Shared Encoder ---
        self.enc1 = nn.Sequential(
            QuantConvBlock(in_channels, 24, kernel=5, padding=2, stride=2), # /2
            QuantConvBlock(24, 24)
        )
        self.enc2 = nn.Sequential(
            QuantConvBlock(24, 48, stride=2), # /4
            QuantConvBlock(48, 48)
        )
        self.enc3 = nn.Sequential(
            QuantConvBlock(48, 96, stride=2), # /8
            QuantConvBlock(96, 96)
        )
        self.bottleneck = nn.Sequential(
            QuantConvBlock(96, 128),
            QuantConvBlock(128, 96)
        )
        self.decoder = nn.Sequential(
            QuantConvBlock(96, 48),
            QuantConvBlock(48, 24)
        )

        # --- Output Head 1: Confidence Map ---
        self.head_confidence = qnn.QuantConv2d(
            24, 1, kernel_size=1, weight_quant=Int8WeightPerTensorFloat, bias=False
        )

        # --- Output Head 2: Theta (Angle) Map ---
        self.head_theta = qnn.QuantConv2d(
            24, num_angles, kernel_size=1, weight_quant=Int8WeightPerTensorFloat, bias=False
        )

        # --- Output Head 3: Regression (w, d) Map ---
        self.head_regression = qnn.QuantConv2d(
            24, num_reg, kernel_size=1, weight_quant=Int8WeightPerTensorFloat, bias=False
        )

    def forward(self, x):
        features = self.enc3(self.enc2(self.enc1(x)))
        features = self.decoder(self.bottleneck(features))

        out_conf = self.head_confidence(features)
        out_theta = self.head_theta(features)
        out_reg = self.head_regression(features)

        return out_conf, out_theta, out_reg