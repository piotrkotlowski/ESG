import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import random

def get_resnet_unet(device="cpu"):
    """
    Phase 3: Stage 2 - Robust Signal Segmentation
    Initialize a U-Net, swapping the standard encoder for a pre-trained ResNet34 backbone
    to improve feature extraction on complex artifacts.
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    return model.to(device)


def apply_heavy_augmentations(image, mask):
    """
    Heavy Augmentation: dynamically inject noise, random synthetic text,
    random vertical lines, and artificially overlay other signal traces.
    """
    # Skeleton placeholder for complex augmentations (e.g. using albumentations)
    # The pipeline aims to force the network to learn trace continuity.
    
    if random.random() > 0.5:
        # Example noise injection
        noise = torch.randn_like(image) * 0.05
        image = torch.clamp(image + noise, 0, 1)
        
    return image, mask


class BCEDiceLoss(nn.Module):
    """
    Loss Function: Combination of Binary Cross-Entropy (BCE) and Dice Loss
    To handle heavy class imbalance (since the trace is only a tiny fraction of the total pixels).
    """
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    def dice_loss(self, pred, target, smooth=1e-5):
        pred = torch.sigmoid(pred)
        # Avoid flattening batch and channel to calculate dice per image
        intersection = (pred * target).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        d_loss = self.dice_loss(pred, target)
        return bce_loss + d_loss
