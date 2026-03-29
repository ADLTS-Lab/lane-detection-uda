import torch
from src.models.unet_resnet import UNetWithResNetEncoder

def test_model_forward():
    model = UNetWithResNetEncoder(n_classes=2)
    x = torch.randn(1, 3, 360, 640)
    out = model(x)
    assert out.shape == (1, 2, 360, 640)