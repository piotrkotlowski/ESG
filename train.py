import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from modeling.dataset import ECGDigitizationDataset

import torch
from modeling.stage1_flattening import DotterUNet
from modeling.stage2_segmentation import get_resnet_unet

def test_model_compilation():
    # 1. Setup device and dummy data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}\n")
    
    # Create a fake batch of images: (Batch_Size, Channels, Height, Width)
    # Using 512x512 as a standard test size for U-Nets
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 512, 512).to(device)
    print(f"Dummy Input Shape: {dummy_input.shape}")

    # -----------------------------------------
    # Test 1: Dotter U-Net (Stage 1)
    # -----------------------------------------
    print("\n--- Testing Dotter U-Net ---")
    try:
        dotter_model = DotterUNet(in_channels=3, out_channels=1).to(device)
        dotter_output = dotter_model(dummy_input)
        
        print(f"SUCCESS! Dotter U-Net Forward Pass Complete.")
        print(f"Expected Output Shape: torch.Size([{batch_size}, 1, 512, 512])")
        print(f"Actual Output Shape:   {dotter_output.shape}")
    except Exception as e:
        print(f"FAILED Dotter U-Net: {e}")

    # -----------------------------------------
    # Test 2: ResNet34 U-Net (Stage 2)
    # -----------------------------------------
    print("\n--- Testing ResNet34 U-Net ---")
    try:
        resnet_model = get_resnet_unet(device=device)
        resnet_output = resnet_model(dummy_input)
        
        print(f"SUCCESS! ResNet34 U-Net Forward Pass Complete.")
        print(f"Expected Output Shape: torch.Size([{batch_size}, 1, 512, 512])")
        print(f"Actual Output Shape:   {resnet_output.shape}")
    except Exception as e:
        print(f"FAILED ResNet34 U-Net: {e}")

if __name__ == "__main__":
    test_model_compilation()

def calculate_metrics(outputs, targets, threshold=0.5, is_logits=False):
    """Calculate Accuracy and F1 Score for batches."""
    if is_logits:
        preds = torch.sigmoid(outputs)
    else:
        preds = outputs
        
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()
    
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    
    return accuracy.item(), f1.item()


    
from modeling.stage1_flattening import DotterUNet
from modeling.stage2_segmentation import get_resnet_unet, apply_heavy_augmentations, BCEDiceLoss

def train_dotter_unet(train_loader, val_loader, device, num_epochs=10):
    """
    Phase 2: Stage 1 - Image Flattening (Dotter U-Net)
    Train the network using raw ECG photos as inputs and dotted grid masks as targets.
    Loss: Mean Squared Error (MSE).
    """
    print("\n--- Starting Training: Dotter U-Net (Stage 1) ---")
    model = DotterUNet(in_channels=3, out_channels=1).to(device)
    
    # From spec.md: Use BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Convert NumPy [B, H, W, C] to PyTorch [B, C, H, W]
            images = batch["image"].permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
            # Grid mask [B, H, W] to [B, 1, H, W]
            targets = batch["grid_mask"].unsqueeze(1).to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        train_loss = running_loss / max(len(train_loader.dataset), 1)
        
        # Validation Loop
        model.eval()
        val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
                targets = batch["grid_mask"].unsqueeze(1).to(device, dtype=torch.float32)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
                acc, f1 = calculate_metrics(outputs, targets, threshold=0.1, is_logits=True)
                val_acc += acc * images.size(0)
                val_f1 += f1 * images.size(0)
                
        val_size = max(len(val_loader.dataset), 1)
        val_loss, val_acc, val_f1 = val_loss / val_size, val_acc / val_size, val_f1 / val_size
        
        print(f"Dotter U-Net -> Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/dotter_unet.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved Dotter U-Net weights to {save_path}")
    return model

def train_resnet_unet(train_loader, val_loader, device, num_epochs=10):
    """
    Phase 3: Stage 2 - Robust Signal Segmentation
    Train using a pre-trained ResNet34 U-Net backbone.
    Loss: Combination of Binary Cross-Entropy (BCE) and Dice Loss.
    """
    print("\n--- Starting Training: ResNet34 U-Net (Stage 2) ---")
    model = get_resnet_unet(device=device)
    
    # From spec.md: Train using a combination of Binary Cross-Entropy (BCE) and Dice Loss
    criterion = BCEDiceLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Prep tensors
            images = batch["image"].permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
            # From spec.md: dataset target_mask is now a Dictionary of masks avoiding overlap.
            # We construct a tensor batch by picking a random lead mask for each item in the batch.
            target_tensors = []
            for i in range(len(batch["image"])):
                # batch["target_mask"] is a dict of lists, or list of dicts depending on default PyTorch collate.
                # PyTorch default_collate turns a list of dicts into a dict of batched lists:
                # { "lead_name": tensor_batch, ... }
                # Let's cleanly grab the first available lead mask to train this forward pass.
                available_leads = list(batch["target_mask"].keys())
                if available_leads:
                    import random
                    chosen_lead = random.choice(available_leads)
                    # Get the mask for this image in the batch
                    mask_i = batch["target_mask"][chosen_lead][i]
                else:
                    # Fallback if no leads exist
                    mask_i = torch.zeros((images.size(2), images.size(3)), dtype=torch.float32)
                target_tensors.append(mask_i)
                
            targets = torch.stack(target_tensors).unsqueeze(1).to(device, dtype=torch.float32)
            
            # From spec.md: Heavy Augmentation: dynamically inject noise, random traces
            images, targets = apply_heavy_augmentations(images, targets)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Loss expects logits, architecture handles this
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        train_loss = running_loss / max(len(train_loader.dataset), 1)
        
        # Validation Loop
        model.eval()
        val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
                # Do the same dictionary extraction for validation targets
                target_tensors_val = []
                for i in range(len(batch["image"])):
                    available_leads = list(batch["target_mask"].keys())
                    if available_leads:
                        mask_i = batch["target_mask"][available_leads[0]][i] # just pick the first for val
                    else:
                        mask_i = torch.zeros((images.size(2), images.size(3)), dtype=torch.float32)
                    target_tensors_val.append(mask_i)
                    
                targets = torch.stack(target_tensors_val).unsqueeze(1).to(device, dtype=torch.float32)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
                acc, f1 = calculate_metrics(outputs, targets, threshold=0.5, is_logits=True)
                val_acc += acc * images.size(0)
                val_f1 += f1 * images.size(0)
                
        val_size = max(len(val_loader.dataset), 1)
        val_loss, val_acc, val_f1 = val_loss / val_size, val_acc / val_size, val_f1 / val_size
        
        print(f"ResNet U-Net -> Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/resnet_unet.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved ResNet34 U-Net weights to {save_path}")
    return model
