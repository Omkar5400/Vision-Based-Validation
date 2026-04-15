import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from my_dataset import AirbagDataset
from tqdm import tqdm
import os

# 1. Hardware Setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"--- Training on: {device.upper()} ---")

# Clear memory cache before starting
if device == "mps":
    torch.mps.empty_cache()

# 2. Model Initialization (ResNet-18 is light and fast)
model = smp.Unet(
    encoder_name="resnet18", 
    encoder_weights="imagenet", 
    in_channels=3, 
    classes=1
)
model.to(device)

# 3. Data & Hyperparameters
IMAGE_PATH = "project-1-at-2026-04-11-12-05-378d87fd/images"
MASK_PATH = "masks"

# Batch size 2 is the 'Sweet Spot' for Mac memory stability
train_ds = AirbagDataset(images_dir=IMAGE_PATH, masks_dir=MASK_PATH)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = smp.losses.DiceLoss(mode='binary')

# 4. Training Loop
num_epochs = 25 
print(f"--- Starting Training for {num_epochs} Epochs ---")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    loop = tqdm(train_loader, total=len(train_loader), leave=True)
    for images, masks in loop:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, masks)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f}")

# 5. Save the final version
torch.save(model.state_dict(), "airbag_unet_v2.pth")
print("\n--- SUCCESS: Optimized Model Saved as 'airbag_unet_v2.pth' ---")