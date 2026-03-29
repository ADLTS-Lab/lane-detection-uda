import torch
import torch.nn as nn
import torchvision.models as models

class UNetWithResNetEncoder(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.n_classes = n_classes

        try:
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        except AttributeError:
            resnet = models.resnet34(pretrained=True)

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc2 = nn.Sequential(resnet.layer1)
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4

        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec5 = self._make_decoder_block(256 + 256, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = self._make_decoder_block(128 + 128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self._make_decoder_block(64 + 64, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _crop_to_match(tensor, target):
        _, _, h, w = tensor.shape
        th, tw = target.shape[2], target.shape[3]
        return tensor[:, :, :th, :tw]

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d5 = self.up5(e5)
        d5 = self._crop_to_match(d5, e4)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = self._crop_to_match(d4, e3)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._crop_to_match(d3, e2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        out = self.up2(d3)
        out = self.final(out)
        return out