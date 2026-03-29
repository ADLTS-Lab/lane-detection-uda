import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

def load_split_txt(file_path, has_labels=True):
    """Reads split file, returns list of (img, mask) if has_labels else list of img."""
    samples = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if has_labels:
                samples.append((parts[0], parts[1]))
            else:
                samples.append(parts[0])
    return samples

class LaneSegmentationDataset(Dataset):
    def __init__(self, root_dir, samples, transform=None, is_labeled=True):
        self.root_dir = root_dir
        self.samples = samples
        self.transform = transform
        self.is_labeled = is_labeled

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_labeled:
            img_path, mask_path = self.samples[idx]
        else:
            img_path = self.samples[idx]
            mask_path = None

        img_full = os.path.join(self.root_dir, img_path)
        img = cv2.imread(img_full)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_full}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if mask_path is not None:
            mask_full = os.path.join(self.root_dir, mask_path)
            mask = cv2.imread(mask_full, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_full}")
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = None

        if self.transform:
            if mask is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=img)
                img = augmented["image"]

        # Ensure img is a tensor
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Ensure mask is a tensor
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.long()
            else:
                mask = torch.from_numpy(mask).long()
        else:
            # For unlabeled, return dummy mask (will be ignored)
            mask = torch.zeros(360, 640, dtype=torch.long)

        # Remove singleton channel if present
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        return img, mask