"""U-Net models.

resnet_unet : U-Net whose contracting path is an ImageNet-pretrained ResNet (18/34/50). It keeps the
              defining U-Net property - an expanding path that concatenates the encoder feature map of
              the same resolution at EVERY level - plus one extra full-resolution skip from the raw RGB input.
vanilla_unet: the original-style U-Net (Ronneberger et al., 2015) trained from scratch (BatchNorm added).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ENC_CHANNELS = {"resnet18": (64, 64, 128, 256, 512),
                "resnet34": (64, 64, 128, 256, 512),
                "resnet50": (64, 256, 512, 1024, 2048)}


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Up(nn.Module):
    """bilinear x2 upsample -> concatenate skip -> two 3x3 convs."""
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = ConvBlock(cin + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNetUNet(nn.Module):
    def __init__(self, encoder="resnet34", pretrained=True, dec=(256, 128, 64, 32, 16)):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        net = getattr(torchvision.models, encoder)(weights=weights)
        c = ENC_CHANNELS[encoder]
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)          # /2
        self.pool, self.l1, self.l2, self.l3, self.l4 = net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4
        self.up4 = Up(c[4], c[3], dec[0])                                  # /16
        self.up3 = Up(dec[0], c[2], dec[1])                                # /8
        self.up2 = Up(dec[1], c[1], dec[2])                                # /4
        self.up1 = Up(dec[2], c[0], dec[3])                                # /2
        self.up0 = Up(dec[3], 3, dec[4])                                   # /1  (skip = raw input)
        self.head = nn.Conv2d(dec[4], 1, 1)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.l1(self.pool(x0))
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        d = self.up4(x4, x3)
        d = self.up3(d, x2)
        d = self.up2(d, x1)
        d = self.up1(d, x0)
        d = self.up0(d, x)
        return self.head(d)

    def encoder_parameters(self):
        mods = [self.stem, self.l1, self.l2, self.l3, self.l4]
        return [p for m in mods for p in m.parameters()]

    def decoder_parameters(self):
        enc = {id(p) for p in self.encoder_parameters()}
        return [p for p in self.parameters() if id(p) not in enc]


class VanillaUNet(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([ConvBlock(3, ch[0])] + [ConvBlock(ch[i], ch[i + 1]) for i in range(4)])
        self.ups = nn.ModuleList([Up(ch[i + 1], ch[i], ch[i]) for i in range(3, -1, -1)])
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        feats = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            feats.append(x)
        x = feats[-1]
        for up, skip in zip(self.ups, feats[-2::-1]):
            x = up(x, skip)
        return self.head(x)

    def encoder_parameters(self):
        return []

    def decoder_parameters(self):
        return list(self.parameters())


def build_model(cfg):
    if cfg["arch"] == "resnet_unet":
        return ResNetUNet(cfg["encoder"], cfg.get("pretrained", True))
    if cfg["arch"] == "vanilla_unet":
        return VanillaUNet(cfg.get("base_channels", 64))
    raise ValueError(f"unknown arch {cfg['arch']}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())
