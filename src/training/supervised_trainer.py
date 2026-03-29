import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.logger import get_logger, success
from torch.utils.tensorboard import SummaryWriter
from .trainer_utils import train_one_epoch, validate

def train_supervised(model, train_loader, val_loader, config, device, save_path, logger=None):
    logger = logger or get_logger()
    epochs = config['training']['num_epochs']
    lr = config['training']['learning_rate']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter(log_dir=config['logging']['log_dir'])

    best_val_loss = float('inf')
    for epoch in range(epochs):
        logger.info("Epoch %d/%d", epoch + 1, epochs)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        logger.info("Train Loss=%.4f | Val Loss=%.4f", train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            success(logger, "Saved best supervised model to %s", save_path)

    writer.close()