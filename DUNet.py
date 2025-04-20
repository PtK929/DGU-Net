import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()


        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print("UP size x11:", x1.size())
        # print("UP size x22:", x2.size())

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

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


class DUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 8)
        self.down1_1 = Down(8, 8)
        self.down1_2 = Down(8, 16)
        self.down1_3 = Down(16, 32)
        self.down1_4 = Down(32, 64)
        self.down1_5 = Down(64, 64)
        self.down2_1 = Down(8, 8)
        self.down2_2 = Down(8, 16)
        self.down2_3 = Down(16, 32)
        self.down2_4 = Down(32, 64)
        self.down2_5 = Down(64, 64)

        self.up1_1 = Up(256,32,bilinear)
        self.up1_2 = Up(64,16,bilinear)
        self.up1_3 = Up(32,8,bilinear)
        self.up1_4 = Up(16,4,bilinear)
        self.up2_1 = Up(256, 32, bilinear)
        self.up2_2 = Up(64, 16, bilinear)
        self.up2_3 = Up(32, 8, bilinear)
        self.up2_4 = Up(16, 4, bilinear)
        self.up5 = Up(16,8,bilinear)

        self.outc = OutConv(8, n_classes)

    def forward(self, x):
        x0 = self.inc(x)
        x1_1 = self.down1_1(x0)
        x1_2 = self.down1_2(x1_1)
        x1_3 = self.down1_3(x1_2)
        x1_4 = self.down1_4(x1_3)
        x1_5 = self.down1_5(x1_4)

        x2_1 = self.down2_1(x0)
        x2_2 = self.down2_2(x2_1)
        x2_3 = self.down2_3(x2_2)
        x2_4 = self.down2_4(x2_3)
        x2_5 = self.down2_5(x2_4)

        x1   = torch.cat([x1_1, x2_1], dim=1)
        x2   = torch.cat([x1_2, x2_2], dim=1)
        x3   = torch.cat([x1_3, x2_3], dim=1)
        x4   = torch.cat([x1_4, x2_4], dim=1)
        x5   = torch.cat([x1_5, x2_5], dim=1)

        x_1 = self.up1_1(x5, x4)
        x_2 = self.up2_1(x5, x4)
        x_1 = self.up1_2(x_1,x1_3)
        x_2 = self.up2_2(x_2, x2_3)
        x_1 = self.up1_3(x_1, x1_2)
        x_2 = self.up2_3(x_2, x2_2)
        x_1 = self.up1_4(x_1, x1_1)
        x_2 = self.up2_4(x_2, x2_1)
        x = torch.cat([x_1, x_2], dim=1)
        x = self.up5(x, x0)

        logits = self.outc(x)
        return logits
