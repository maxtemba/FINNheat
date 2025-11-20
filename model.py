import torch
import torch.nn as nn
import brevitas.nn as qnn
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
            return_quant_tensor=False
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FINNCompatibleGHM_MultiOutput(nn.Module):
    def __init__(self, in_channels=4, num_angles=6):
        super().__init__()

        # ======================================================================
        # 1. INPUT QUANTIZATION (Crucial for FINN)
        # ======================================================================
        self.quant_input = qnn.QuantIdentity(
            bit_width=8,
            return_quant_tensor=True,
            act_quant=Int8ActPerTensorFloat
        )

        # ======================================================================
        # 2. BACKBONE (Encoder + Decoder)
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

        # ======================================================================
        # 3. HGGD-COMPATIBLE HEADS (5 Outputs)
        # ======================================================================
        # 1. Location Map (Confidence) -> 1 Channel
        self.head_loc = qnn.QuantConv2d(
            32, 1, kernel_size=1,
            weight_quant=Int8WeightPerTensorFloat, bias=True
        )

        # 2. Classification Mask (Anchor Confidence) -> K Channels
        self.head_cls = qnn.QuantConv2d(
            32, num_angles, kernel_size=1,
            weight_quant=Int8WeightPerTensorFloat, bias=True
        )

        # 3. Theta Offset Regression -> K Channels
        self.head_theta_off = qnn.QuantConv2d(
            32, num_angles, kernel_size=1,
            weight_quant=Int8WeightPerTensorFloat, bias=True
        )

        # 4. Width Offset Regression -> K Channels
        self.head_width_off = qnn.QuantConv2d(
            32, num_angles, kernel_size=1,
            weight_quant=Int8WeightPerTensorFloat, bias=True
        )

        # 5. Depth Offset Regression -> K Channels
        self.head_depth_off = qnn.QuantConv2d(
            32, num_angles, kernel_size=1,
            weight_quant=Int8WeightPerTensorFloat, bias=True
        )

    def forward(self, x):
        # 1. Quantize Input
        x = self.quant_input(x)

        # 2. Backbone
        x = self.enc1(x)   # -> H/2
        x = self.enc2(x)   # -> H/4
        x = self.enc3(x)   # -> H/8
        feat = self.decoder(self.bottleneck(x))

        # 3. Heads
        loc_map = self.head_loc(feat)
        cls_mask = self.head_cls(feat)
        theta_off = self.head_theta_off(feat)
        width_off = self.head_width_off(feat)
        depth_off = self.head_depth_off(feat)

        return loc_map, cls_mask, theta_off, width_off, depth_off