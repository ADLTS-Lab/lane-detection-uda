import torch
from src.utils.metrics import compute_iou

def test_iou():
    pred = torch.tensor([[0,1],[1,0]])
    target = torch.tensor([[0,1],[1,0]])
    ious = compute_iou(pred, target, num_classes=2)
    assert ious[0] == 1.0 and ious[1] == 1.0