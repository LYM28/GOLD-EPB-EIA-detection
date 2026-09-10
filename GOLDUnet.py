import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.3):  # 添加dropout_rate参数
        super().__init__()
        self.dropout_rate = dropout_rate  # 保存dropout率

        # 定义带dropout的卷积块
        def CBR(in_ch, out_ch, apply_dropout=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            # 根据需要添加dropout层
            if apply_dropout:
                layers.append(nn.Dropout2d(self.dropout_rate))  # 使用2D dropout更适合卷积层
            return nn.Sequential(*layers)

        # 编码器部分（添加dropout）
        self.enc1 = nn.Sequential(
            CBR(3, 64),
            CBR(64, 64)
        )
        self.enc2 = nn.Sequential(
            CBR(64, 128),
            CBR(128, 128)
        )
        self.enc3 = nn.Sequential(
            CBR(128, 256),
            CBR(256, 256)
        )
        self.enc4 = nn.Sequential(
            CBR(256, 512),
            CBR(512, 512)
        )

        self.pool = nn.MaxPool2d(2)

        # 中心部分（添加dropout）
        self.center = nn.Sequential(
            CBR(512, 1024),
            CBR(1024, 1024)
        )

        # 解码器部分（添加dropout）
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = nn.Sequential(
            CBR(1024, 512),
            CBR(512, 512)
        )

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            CBR(512, 256),
            CBR(256, 256)
        )

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            CBR(256, 128),
            CBR(128, 128)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            CBR(128, 64),
            CBR(64, 64)
        )

        # 输出层不添加dropout
        self.out_conv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        center = self.center(self.pool(enc4))

        dec4 = self.dec4(torch.cat([self.up4(center), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        return self.out_conv(dec1)
