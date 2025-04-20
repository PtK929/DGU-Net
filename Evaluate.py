import torch
import torch.nn.functional as F
from tqdm import tqdm

from Dice_score import multiclass_dice_coeff, dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)

    dice_score = 0.0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    autocast_device = device.type if device.type != 'mps' else 'cpu'
    with torch.autocast(autocast_device, enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)


            mask_pred = net(image)

            if net.n_classes == 1:
                mask_prob = torch.sigmoid(mask_pred)  # [B,1,H,W]
                mask_pred_bin = (mask_prob > 0.5).float()  # [B,1,H,W]

                mask_true_bin = mask_true.squeeze(1).float()
                mask_pred_bin = mask_pred_bin.squeeze(1)


                dice_score += dice_coeff(mask_pred_bin, mask_true_bin, reduce_batch_first=False)
                pred_bool = (mask_pred_bin == 1)
                true_bool = (mask_true_bin == 1)

                TP += torch.sum(pred_bool & true_bool).item()
                TN += torch.sum(~pred_bool & ~true_bool).item()
                FP += torch.sum(pred_bool & ~true_bool).item()
                FN += torch.sum(~pred_bool & true_bool).item()
            else:
                mask_true_hot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_hot = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred_hot[:, 1:], mask_true_hot[:, 1:], reduce_batch_first=False)

    net.train()

    dice_value = dice_score / max(num_val_batches, 1)

    if net.n_classes == 1:
        epsilon = 1e-7
        JA = TP / (TP + FP + FN + epsilon)
        AC = (TP + TN) / (TP + TN + FP + FN + epsilon)
        SE = TP / (TP + FN + epsilon)
        SP = TN / (TN + FP + epsilon)

        return {
            'Dice': dice_value.item(),
            'JA': JA,
            'AC': AC,
            'SE': SE,
            'SP': SP
        }
    else:
        return {
            'Dice': dice_value.item()
        }
