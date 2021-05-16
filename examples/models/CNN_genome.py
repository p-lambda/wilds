import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def single_conv(in_channels, out_channels, kernel_size=7):
    padding_size = int((kernel_size-1)/2)
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding_size),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    )

def double_conv(in_channels, out_channels, kernel_size=7):
    padding_size = int((kernel_size-1)/2)
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding_size),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding_size),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, num_tasks=16, n_channels_in=5):
        super().__init__()
        
        self.dconv_down1 = double_conv(n_channels_in, 15)
        self.dconv_down2 = double_conv(15, 22)
        self.dconv_down3 = double_conv(22, 33)
        self.dconv_down4 = double_conv(33, 49)
        self.dconv_down5 = double_conv(49, 73)
        self.dconv_down6 = double_conv(73, 109)

        self.maxpool = nn.MaxPool1d(2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv_middle = single_conv(109, 109)
        self.upsamp_6 = nn.ConvTranspose1d(109, 109, 2, stride=2)

        self.dconv_up5 = double_conv(73 + 109, 73)
        self.upsamp_5 = nn.ConvTranspose1d(73, 73, 2, stride=2)
        self.dconv_up4 = double_conv(49 + 73, 49)
        self.upsamp_4 = nn.ConvTranspose1d(49, 49, 2, stride=2)
        self.dconv_up3 = double_conv(33 + 49, 33)
        self.upsamp_3 = nn.ConvTranspose1d(33, 33, 2, stride=2)
        self.dconv_up2 = double_conv(22 + 33, 22)
        self.upsamp_2 = nn.ConvTranspose1d(22, 22, 2, stride=2)
        self.dconv_up1 = double_conv(15 + 22, 15)
        self.upsamp_1 = nn.ConvTranspose1d(15, 15, 2, stride=2)

        self.conv_last = nn.Conv1d(15, 1, 200, stride=50, padding=0)
        self.d_out = num_tasks if num_tasks is not None else 253
        
        self.fc_last = nn.Linear(253, 128)


    def forward(self, x):
        # input_size = 12800
        # input_channels = 5
        x = x.float()
        conv1 = self.dconv_down1(x)     # Output size: (input_size) x 15
        x = self.maxpool(conv1)         # (input_size / 2) x 15

        conv2 = self.dconv_down2(x)     # (input_size / 2) x 22
        x = self.maxpool(conv2)         # (input_size / 4) x 22

        conv3 = self.dconv_down3(x)     # (input_size / 4) x 33
        x = self.maxpool(conv3)         # (input_size / 8) x 33

        conv4 = self.dconv_down4(x)     # (input_size / 8) x 49
        x = self.maxpool(conv4)         # (input_size / 16) x 49

        conv5 = self.dconv_down5(x)     # (input_size / 16) x 73
        x = self.maxpool(conv5)         # (input_size / 32) x 73

        conv6 = self.dconv_down6(x)     # (input_size / 32) x 109
        # conv6 = self.conv_middle(conv6)  # Optional: convolution here.

        # Encoder finished.

        x = self.upsamp_6(conv6)          # (input_size / 16) x 109
        x = torch.cat([x, conv5], dim=1)  # (input_size / 16) x (109 + 73)

        x = self.dconv_up5(x)             # (input_size / 16) x 73
        x = self.upsamp_5(x)              # (input_size / 8) x 73
        x = torch.cat([x, conv4], dim=1)  # (input_size / 8) x (73 + 49)

        x = self.dconv_up4(x)             # (input_size / 8) x 49
        x = self.upsamp_4(x)              # (input_size / 4) x 49
        x = torch.cat([x, conv3], dim=1)  # (input_size / 4) x (49 + 33)

        x = self.dconv_up3(x)             # (input_size / 4) x 33
        x = self.upsamp_3(x)              # (input_size / 2) x 33
        x = torch.cat([x, conv2], dim=1)  # (input_size / 2) x (33 + 22)

        x = self.dconv_up2(x)             # (input_size / 2) x 22
        x = self.upsamp_2(x)              # (input_size) x 22
        x = torch.cat([x, conv1], dim=1)  # (input_size) x (22 + 15)

        x = self.dconv_up1(x)             # (input_size) x 15

        x = self.conv_last(x)             # (input_size/50 - 3) x 1
        x = torch.squeeze(x)
        
        # Default input_size == 12800: x has size N x 1 x 253 at this point.
        if self.d_out == 253:
            out = x
        else:
            out = self.fc_last(x)
            # out = x[:, 64:192]    # middle 128 values
        
        return out
