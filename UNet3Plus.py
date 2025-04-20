# newTestUNet3plus.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3Plus(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet3Plus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.conv1 = ConvBlock(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = ConvBlock(512, 1024)

        self.up1 = UpConv(1024 + 512, 512, bilinear)  # 1536 -> 512
        self.up2 = UpConv(512 + 256, 256, bilinear)   # 768 -> 256
        self.up3 = UpConv(256 + 128, 128, bilinear)   # 384 -> 128
        self.up4 = UpConv(128 + 64, 64, bilinear)     # 192 -> 64

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        c1 = self.conv1(x)   
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)  
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)  
        p3 = self.pool3(c3)
        c4 = self.conv4(p3) 
        p4 = self.pool4(c4)
        c5 = self.conv5(p4) 


        u1 = self.up1(c5, c4)  
        u2 = self.up2(u1, c3)  
        u3 = self.up3(u2, c2)  
        u4 = self.up4(u3, c1)  

        logits = self.outc(u4)
        return logits

