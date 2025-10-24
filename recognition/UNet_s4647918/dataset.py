import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class OASIS2DDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert("L")  # grayscale
        mask = Image.open(mask_path).convert("L")  # single-channel mask

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


#Visualisation code to check this works as expected

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = OASIS2DDataset(
    image_dir="C:/Users/ch2ck/OASIS_full/keras_png_slices_train",
    mask_dir="C:/Users/ch2ck/OASIS_full/keras_png_slices_seg_train",
    transform=transform,
    target_transform=transform
)


    print(f"Total samples in dataset: {len(dataset)}")

    image, mask = dataset[0]
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask.squeeze(), cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

