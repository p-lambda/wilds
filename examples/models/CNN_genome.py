import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def single_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 7, padding=3), 
        nn.BatchNorm1d(out_channels), 
        nn.ReLU(inplace=True)
    )

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 7, padding=3), 
        nn.BatchNorm1d(out_channels), 
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, 7, padding=3), 
        nn.BatchNorm1d(out_channels), 
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, n_class, n_channels_in=6):
        super().__init__()
        
        self.dconv_down1 = double_conv(n_channels_in, 15)
        self.dconv_down2 = double_conv(15, 22)
        self.dconv_down3 = double_conv(22, 33)
        self.dconv_down4 = double_conv(33, 49)
        self.dconv_down5 = double_conv(49, 73)
        self.dconv_down6 = double_conv(73, 109)

        self.maxpool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_middle = single_conv(109, 109)
        
        self.dconv_up5 = double_conv(73 + 109, 73)
        self.dconv_up4 = double_conv(49 + 73, 49)
        self.dconv_up3 = double_conv(33 + 49, 33)
        self.dconv_up2 = double_conv(22 + 33, 22)
        self.dconv_up1 = double_conv(15 + 22, 15)
        
        self.conv_last = nn.Conv2d(15, n_class, 1)
        
        
    def forward(self, x):
        # input_size = 12800
        # input_channels = 6
        conv1 = self.dconv_down1(x)     # Out: (input_size) x 15
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
        
        x = self.upsample(conv6)          # (input_size / 16) x 109
        x = torch.cat([x, conv5], dim=1)  # (input_size / 16) x (109 + 73)
        
        x = self.dconv_up5(x)             # (input_size / 16) x 73
        x = self.upsample(x)              # (input_size / 8) x 73
        x = torch.cat([x, conv4], dim=1)  # (input_size / 8) x (73 + 49)
        
        x = self.dconv_up4(x)             # (input_size / 8) x 49
        x = self.upsample(x)              # (input_size / 4) x 49
        x = torch.cat([x, conv3], dim=1)  # (input_size / 4) x (49 + 33)
        
        x = self.dconv_up3(x)             # (input_size / 4) x 33
        x = self.upsample(x)              # (input_size / 2) x 33
        x = torch.cat([x, conv2], dim=1)  # (input_size / 2) x (33 + 22)

        x = self.dconv_up2(x)             # (input_size / 2) x 22
        x = self.upsample(x)              # (input_size) x 22
        x = torch.cat([x, conv1], dim=1)  # (input_size) x (22 + 15)
        
        x = self.dconv_up1(x)             # (input_size) x 15
        
        out = self.conv_last(x)
        
        return out
