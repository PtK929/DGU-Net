import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,  sigma=1.0):
        super(GaussianConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # self.padding = padding

        self.sigma = sigma

        self.padding = (kernel_size - 1) // 2

        self.create_gaussian_kernel()

    def create_gaussian_kernel(self):

        x_cord = torch.arange(self.kernel_size).float()
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (self.kernel_size - 1) / 2.
        variance = self.sigma ** 2.


        gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
        )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)


        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.out_channels, self.in_channels, 1, 1)

        self.gaussian_kernel = nn.Parameter(gaussian_kernel, requires_grad=False)

    def forward(self, input):
        # 应用高斯卷积        return F.conv2d(input, self.gaussian_kernel, stride=self.stride, padding=self.padding)


class GaussianDoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gaussian_conv = GaussianConv2d(in_channels, in_channels, kernel_size=9, sigma=1.0)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.gaussian_conv(x)
        x = self.double_conv(x)
        return x




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


class GaussianDown(nn.Module):


    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gaussian_conv = GaussianConv2d(in_channels, in_channels, kernel_size=9, sigma=1.0)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.gaussian_conv(x)
        x = self.maxpool_conv(x)
        return x



class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class GaussianUp(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.gaussian_conv = GaussianConv2d(in_channels//2, in_channels//2, kernel_size=9, sigma=1.0)

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.gaussian_conv(x1)

        x1 = self.up(x1)
        # print("UP size x1:", x1.size())
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

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


class GUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(GUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = GaussianDoubleConv(n_channels, 64 // 8)
        self.down1 = GaussianDown(64 // 8, 128 // 8)
        self.down2 = Down(128 // 8, 256 // 8)
        self.down3 = Down(256 // 8, 512 // 8)
        self.down4 = Down(512 // 8, 1024 // 8)
        self.down5 = GaussianDown(1024 // 8, 1024 // 8)
        self.up1 = GaussianUp(2048 // 8, 512 // 8, bilinear)
        self.up2 = Up(1024 // 8, 256 // 8, bilinear)
        self.up3 = Up(512 // 8, 128 // 8, bilinear)
        self.up4 = Up(256 // 8, 64 // 8, bilinear)
        self.up5 = GaussianUp(128 // 8, 64 // 8, bilinear)
        self.outc = OutConv(64 // 8, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits
