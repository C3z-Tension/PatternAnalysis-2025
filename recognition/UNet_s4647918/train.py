import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import ImprovedUNet, DiceLoss
from dataset import OASIS2DDataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

# Base directory (can be set via environment variable or config file)
base_dir = os.getenv("OASIS_DATA_DIR", "./OASIS_full")


# -----------------------------
# Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Small training size as I don't have an NVIDIA Graphics card
epochs = 10
lr = 1e-4
batch_size = 2
image_dir = os.path.join(base_dir, "keras_png_slices_train")
mask_dir = os.path.join(base_dir, "keras_png_slices_seg_train")
save_path = os.path.join(base_dir, "improved_unet.pth")

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# -----------------------------
# Dataset and DataLoaders
# -----------------------------
full_dataset = OASIS2DDataset(image_dir, mask_dir, transform=transform, target_transform=transform)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

#Split into desired sizes and allocate to prevent data leakage
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

# -----------------------------
# Model, Loss, Optimizer
# -----------------------------
model = ImprovedUNet().to(device)
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# -----------------------------
# Training Loop
# -----------------------------
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

# -----------------------------
# Save Model
# -----------------------------
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

# -----------------------------
# Plot Losses
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("loss_plot.png")
plt.show()

# -----------------------------
# Testing
# -----------------------------
model.eval()
test_loss = 0.0

with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")