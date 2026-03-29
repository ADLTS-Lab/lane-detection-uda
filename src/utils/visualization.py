import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def denormalize(image_tensor):
    """Denormalize image from ImageNet stats."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image_tensor.cpu().numpy().transpose(1,2,0)
    image = image * std + mean
    return np.clip(image, 0, 1)

def save_prediction_overlay(image_tensor, pred_mask, true_mask, output_path):
    """Save side-by-side comparison: image, ground truth, prediction."""
    img = denormalize(image_tensor)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis('off')
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title("Prediction")
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()