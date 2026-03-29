import torch

def compute_iou(preds, targets, num_classes=2):
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou = torch.tensor(float('nan'), device=preds.device)
        else:
            iou = intersection / union
        ious.append(iou)
    return ious

def compute_f1(preds, targets, num_classes=2):
    """F1 score per class."""
    f1s = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        tp = (pred_inds & target_inds).sum().float()
        fp = (pred_inds & ~target_inds).sum().float()
        fn = (~pred_inds & target_inds).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1)
    return f1s

def compute_brier_score(probs, targets):
    """Brier score = mean squared error between predicted probabilities and one-hot labels."""
    # probs: (N, C, H, W), targets: (N, H, W)
    one_hot = torch.nn.functional.one_hot(targets, num_classes=probs.shape[1]).permute(0,3,1,2).float()
    brier = ((probs - one_hot) ** 2).mean().item()
    return brier

def compute_fp_fn(preds, targets, num_classes=2):
    """Return false positive and false negative rates per class."""
    fp_rates = []
    fn_rates = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        fp = (pred_inds & ~target_inds).sum().float()
        fn = (~pred_inds & target_inds).sum().float()
        total_pos = target_inds.sum().float()
        if total_pos > 0:
            fp_rate = fp / total_pos
            fn_rate = fn / total_pos
        else:
            fp_rate = torch.tensor(float('nan'), device=preds.device)
            fn_rate = torch.tensor(float('nan'), device=preds.device)
        fp_rates.append(fp_rate)
        fn_rates.append(fn_rate)
    return fp_rates, fn_rates

def compute_pixel_entropy(logits):
    """Return average pixel entropy over the batch."""
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # per pixel
    return entropy.mean().item()   # average over all pixels and batch