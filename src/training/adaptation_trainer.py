import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.utils.logger import get_logger, success
from .trainer_utils import entropy_loss, validate

def train_adaptation(model, src_loader, tgt_loader, val_loader, config, device, save_path, logger=None):
    logger = logger or get_logger()
    epochs = config['training']['num_epochs']
    alpha = config['uda']['alpha']
    lr = config['uda']['learning_rate']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_sup_loss = 0.0
        total_ent_loss = 0.0
        num_batches = min(len(src_loader), len(tgt_loader))
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)

        pbar = tqdm(
            range(num_batches),
            desc=f"Adaptation Epoch {epoch + 1}/{epochs}",
            leave=False,
            colour="green",
        )
        for _ in pbar:
            try:
                src_images, src_masks = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                src_images, src_masks = next(src_iter)
            try:
                tgt_images, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_images, _ = next(tgt_iter)

            src_images = src_images.to(device)
            src_masks = src_masks.to(device)
            if src_masks.dim() == 4 and src_masks.shape[1] == 1:
                src_masks = src_masks.squeeze(1)
            tgt_images = tgt_images.to(device)

            optimizer.zero_grad()
            src_logits = model(src_images)
            sup_loss = criterion(src_logits, src_masks)

            tgt_logits = model(tgt_images)
            ent_loss = entropy_loss(tgt_logits)

            loss = sup_loss + alpha * ent_loss
            loss.backward()
            optimizer.step()

            total_sup_loss += sup_loss.item()
            total_ent_loss += ent_loss.item()

            pbar.set_postfix(sup_loss=sup_loss.item(), ent_loss=ent_loss.item())

        logger.info(
            "Epoch %d/%d | Sup Loss=%.4f | Ent Loss=%.4f",
            epoch + 1,
            epochs,
            total_sup_loss / num_batches,
            total_ent_loss / num_batches,
        )

        val_loss = validate(model, val_loader, criterion, device)
        logger.info("Target Val Loss=%.4f", val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            success(logger, "Saved best adapted model to %s", save_path)

        scheduler.step()