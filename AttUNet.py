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


class AttentionGate(nn.Module):


    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        diffY = x1.size()[2] - g1.size()[2]
        diffX = x1.size()[3] - g1.size()[3]

        if diffY != 0 or diffX != 0:
            g1 = F.pad(g1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpConv(nn.Module):


    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:

            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        else:
            
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


        self.att = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)

        self.conv = ConvBlock(2 * out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        att_x2 = self.att(x1, x2)
        x = torch.cat([x2, att_x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AttUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(AttUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ConvBlock(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = ConvBlock(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(512, 1024)


        self.up1 = UpConv(1024, 512, bilinear)
        self.up2 = UpConv(512, 256, bilinear)
        self.up3 = UpConv(256, 128, bilinear)
        self.up4 = UpConv(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):

        x1 = self.inc(x)  
        x2 = self.down1(x1)  #
        x2 = self.conv1(x2)  
        x3 = self.down2(x2)  
        x3 = self.conv2(x3)  
        x4 = self.down3(x3) 
        x4 = self.conv3(x4) 
        x5 = self.down4(x4) 
        x5 = self.conv4(x5)  

        # 解码器
        u1 = self.up1(x5, x4)  
        u2 = self.up2(u1, x3)  
        u3 = self.up3(u2, x2)  
        u4 = self.up4(u3, x1)  

        logits = self.outc(u4)  
        return logits



