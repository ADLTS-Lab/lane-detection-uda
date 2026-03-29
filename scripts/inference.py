#!/usr/bin/env python
import argparse
import cv2
import torch
import numpy as np
from src.models.unet_resnet import UNetWithResNetEncoder
from src.data.transforms import get_val_transform
from src.utils.logger import setup_logger, section, success

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to .pth model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='output.png', help='Output path')
    args = parser.parse_args()
    logger = setup_logger()
    section(logger, "Running lane inference")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetWithResNetEncoder(n_classes=2).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = get_val_transform(360, 640)
    transformed = transform(image=img)
    img_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Overlay on original image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay = np.zeros_like(img)
    overlay[pred == 1] = [0, 255, 0]  # green for lane
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    cv2.imwrite(args.output, blended)
    success(logger, "Saved prediction to %s", args.output)

if __name__ == "__main__":
    main()