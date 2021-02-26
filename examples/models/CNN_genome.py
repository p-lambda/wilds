import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



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

    def __init__(self, n_class):
        super().__init__()
        
        self.dconv_down1 = double_conv(6, 15)
        self.dconv_down2 = double_conv(15, 22)
        self.dconv_down3 = double_conv(22, 33)
        self.dconv_down4 = double_conv(33, 49)
        self.dconv_down5 = double_conv(49, 73)
        self.dconv_down6 = double_conv(73, 109)

        self.maxpool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up5 = double_conv(73 + 109, 73)
        self.dconv_up4 = double_conv(49 + 73, 49)
        self.dconv_up3 = double_conv(33 + 49, 33)
        self.dconv_up2 = double_conv(22 + 33, 22)
        self.dconv_up1 = double_conv(15 + 22, 15)
        
        self.conv_last = nn.Conv2d(15, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        
        conv5 = self.dconv_down5(x)
        x = self.maxpool(conv5)
        
        x = self.dconv_down6(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv5], dim=1)
        
        x = self.dconv_up5(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv4], dim=1)
        
        x = self.dconv_up4(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
