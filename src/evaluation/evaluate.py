import torch
import numpy as np
from tqdm import tqdm
import json
import os
from pathlib import Path
from src.utils.metrics import compute_iou, compute_f1, compute_brier_score, compute_fp_fn, compute_pixel_entropy
from src.utils.visualization import save_prediction_overlay
from src.utils.logger import get_logger

def evaluate_model(model, dataloader, device, output_dir=None, tag='model', logger=None):
    logger = logger or get_logger()
    model.eval()
    total_iou = 0.0
    total_f1 = 0.0
    total_brier = 0.0
    total_entropy = 0.0
    total_fp = 0.0
    total_fn = 0.0
    num_samples = 0

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, (images, masks) in enumerate(
            tqdm(dataloader, desc=f"Evaluating {tag}", colour="green")
        ):
            images = images.to(device)
            masks = masks.to(device)
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Compute metrics
            ious = compute_iou(preds, masks, num_classes=2)
            lane_iou = ious[1] if not torch.isnan(ious[1]) else torch.tensor(0.0)
            total_iou += lane_iou.item() * images.size(0)

            f1s = compute_f1(preds, masks, num_classes=2)
            total_f1 += f1s[1].item() * images.size(0)  # lane class

            probs = torch.softmax(outputs, dim=1)
            brier = compute_brier_score(probs, masks)
            total_brier += brier * images.size(0)

            entropy = compute_pixel_entropy(outputs)  # average over batch
            total_entropy += entropy * images.size(0)

            fp_rates, fn_rates = compute_fp_fn(preds, masks, num_classes=2)
            total_fp += fp_rates[1].item() * images.size(0)
            total_fn += fn_rates[1].item() * images.size(0)

            num_samples += images.size(0)

            # Save visualizations for a few samples
            if output_dir and i < 4:  # first 4 batches
                for j in range(min(4, images.size(0))):
                    save_path = output_dir / f"{tag}_sample_{i}_{j}.png"
                    save_prediction_overlay(images[j].cpu(), preds[j].cpu(), masks[j].cpu(), save_path)

    metrics = {
        'iou': total_iou / num_samples,
        'f1': total_f1 / num_samples,
        'brier': total_brier / num_samples,
        'pixel_entropy': total_entropy / num_samples,
        'fp_rate': total_fp / num_samples,
        'fn_rate': total_fn / num_samples,
    }
    logger.info(
        "Evaluation %s | IoU=%.4f | F1=%.4f | Brier=%.4f | Entropy=%.4f | FP=%.4f | FN=%.4f",
        tag,
        metrics['iou'],
        metrics['f1'],
        metrics['brier'],
        metrics['pixel_entropy'],
        metrics['fp_rate'],
        metrics['fn_rate'],
    )
    return metrics