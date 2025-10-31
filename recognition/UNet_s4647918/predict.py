import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from modules import ImprovedUNet
from dataset import OASIS2DDataset

# -----------------------------
# Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "C:/Users/ch2ck/OASIS_full/improved_unet.pth"
image_dir = "C:/Users/ch2ck/OASIS_full/keras_png_slices_train"
mask_dir = "C:/Users/ch2ck/OASIS_full/keras_png_slices_seg_train"

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# -----------------------------
# Load Dataset
# -----------------------------
dataset = OASIS2DDataset(image_dir, mask_dir, transform=transform, target_transform=transform)
sample_idx = 0  # Change this to test different samples
image, mask = dataset[sample_idx]
image = image.unsqueeze(0).to(device)  # Add batch dimension

# -----------------------------
# Load Model
# -----------------------------
model = ImprovedUNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -----------------------------
# Run Prediction
# -----------------------------
with torch.no_grad():
    output = model(image)
    pred_mask = output.squeeze().cpu().numpy()

# -----------------------------
# Visualize Results
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image.squeeze().cpu(), cmap="gray")
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask.squeeze().cpu(), cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pred_mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.tight_layout()
plt.show()
