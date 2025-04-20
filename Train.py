import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from Evaluate import evaluate

from Data import CustomDataset
from Dice_score import dice_loss

from DGUNet import DGUNet



dir_img = Path()
dir_mask = Path()
dir_checkpoint = Path()

def train_model(
        model,
        device,
        epochs: int = 80,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.2,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.9999,
        gradient_clipping: float = 1.0,
):

    dataset = CustomDataset(dir_img, dir_mask)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, val_percent=val_percent, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}

        Device:          {device.type}

        Mixed Precision: {amp}
    ''')

    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    best_dice = 0.0
    best_epoch = 0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels,' \
                    f' but loaded images have {images.shape[1]} channels.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)

                    
                    true_masks = true_masks.squeeze(1)
                    loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})


                division_step = (n_train // (5 * batch_size))
                if division_step > 0 and global_step % division_step == 0:
                    histograms = {}
                    for tag, value in model.named_parameters():
                        tag = tag.replace('/', '.')
                        if value is not None and value.data is not None and not (torch.isinf(value.data) | torch.isnan(value.data)).any():
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())


                    val_metrics = evaluate(model, val_loader, device, amp)

                    
                    scheduler.step(val_metrics['Dice'])
                    

                    logging.info(f'Validation metrics: {val_metrics}')

 
                    current_dice = val_metrics['Dice']
                    if current_dice > best_dice:
                        best_dice = current_dice
                        best_epoch = epoch
                        best_metrics = val_metrics.copy()


 
                    log_dict = {
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'validation Dice': val_metrics['Dice'],
                        'images': wandb.Image(images[0].cpu()),
                        'masks': {
                            'true': wandb.Image(true_masks[0].float().cpu()),
                        },
                        'step': global_step,
                        'epoch': epoch,
                        **histograms
                    }


                    if 'JA' in val_metrics:
                        log_dict['validation JA'] = val_metrics['JA']
                    if 'AC' in val_metrics:
                        log_dict['validation AC'] = val_metrics['AC']
                    if 'SE' in val_metrics:
                        log_dict['validation SE'] = val_metrics['SE']
                    if 'SP' in val_metrics:
                        log_dict['validation SP'] = val_metrics['SP']


                    pred_mask = (torch.sigmoid(masks_pred) > 0.5).float()
                    log_dict['masks']['pred'] = wandb.Image(pred_mask[0].cpu())

                    experiment.log(log_dict)

    if save_checkpoint:
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        state_dict['mask_values'] = dataset.mask_values
        # torch.save(state_dict, str(dir_checkpoint / f'checkpoint_GaussStartEnd_epoch{epoch}.pth'))
        torch.save(state_dict, str(dir_checkpoint / f'checkpoint_DGUNet_epoch{epoch}.pth'))
        logging.info(f'Checkpoint {epoch} saved!')


    logging.info(f"Best Dice: {best_dice} at epoch {best_epoch}")
    logging.info(f"Best validation metrics at epoch {best_epoch}: {best_metrics}")


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = DGUNet(n_channels=3, n_classes=1, bilinear=True)

    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    # )

    train_model(
        model=model,
        epochs=80,
        batch_size=2,
        learning_rate=5e-5,
        device=device,
        val_percent=20 / 100,
        amp=False
    )
