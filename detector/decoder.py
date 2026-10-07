import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)



class Decoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.first_conv = nn.Conv2d(in_channels+2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.feat2d = 512

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingConcat(256 + 128, 256)
        self.up2_skip = UpsamplingConcat(256 + 64, 256)
        # self.up1_skip = UpsamplingConcat(256 + 512, shared_out_channels)
        self.up1_skip = UpsamplingConcat(256 + 514, shared_out_channels)

        
    # def forward(self, x, feat_cams):
    def forward(self, x):
        b, c, h, w = x.shape

        # pad input
        m = 8
        ph, pw = math.ceil(h / m) * m - h, math.ceil(w / m) * m - w
        pt, pb = ph // 2, ph - (ph // 2)
        pl, pr = pw // 2, pw - (pw // 2)
        x = torch.nn.functional.pad(x, (pl, pr, pt, pb))

        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        # Unpad
        x = x[..., pt:pt + h, pl:pl + w]

        # Extra upsample to (2xH, 2xW)
        # x = self.up_sample_2x(x)

        return x
