import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two sequential 3D Conv → InstanceNorm → ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Strided 2×2×2 conv (stride=2) → ConvBlock."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv3d(in_ch, in_ch, kernel_size=2, stride=2, bias=False)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.down(x))


class UpBlock(nn.Module):
    """Transposed conv upsampling + skip-connection concat + ConvBlock."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Align spatial size if zoom/pool leaves a 1-voxel mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='trilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net for BraTS multi-class segmentation.

    Input : (B, 4,  128, 128, 128)
    Output: (B, 4,  128, 128, 128)  — raw logits (4 classes)
    """

    def __init__(self, in_channels=4, num_classes=4, base_ch=32):
        super().__init__()
        f = base_ch  # 32 by default

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)       # → (B, 32, 128,128,128)
        self.enc2 = DownBlock(f,     f * 2)         # → (B, 64,   64, 64, 64)
        self.enc3 = DownBlock(f * 2, f * 4)         # → (B,128,   32, 32, 32)
        self.enc4 = DownBlock(f * 4, f * 8)         # → (B,256,   16, 16, 16)

        # Bottleneck
        self.bottleneck = DownBlock(f * 8, f * 16)  # → (B,512,    8,  8,  8)

        # Decoder
        self.dec4 = UpBlock(f * 16, f * 8, f * 8)  # → (B,256,   16, 16, 16)
        self.dec3 = UpBlock(f * 8,  f * 4, f * 4)  # → (B,128,   32, 32, 32)
        self.dec2 = UpBlock(f * 4,  f * 2, f * 2)  # → (B, 64,   64, 64, 64)
        self.dec1 = UpBlock(f * 2,  f,     f)       # → (B, 32,  128,128,128)

        # 1×1×1 projection to num_classes
        self.out_conv = nn.Conv3d(f, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.out_conv(d1)
