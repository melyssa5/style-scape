import os
import random
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

class StylizedPairDataset(Dataset):
    """
    Custom dataset to return a pair: (natural_image, stylized_variant), with optional transform.
    Assumes stylized dataset contains exactly 3x the number of images as the natural one.
    """
    def __init__(self, natural_root, stylized_root, transform=None):
        self.natural_data = ImageFolder(natural_root)
        self.stylized_data = ImageFolder(stylized_root)
        self.transform = transform

        assert len(self.stylized_data) == len(self.natural_data) * 3, \
            f"Expected 3 stylized variants per natural image. Got {len(self.stylized_data)} stylized and {len(self.natural_data)} natural."

    def __getitem__(self, idx):
        # Get the natural image
        img1, _ = self.natural_data[idx]

        # Choose one of the 3 stylized variants
        stylized_idx = idx * 3 + random.randint(0, 2)
        img2, _ = self.stylized_data[stylized_idx]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), 0  # dummy label

    def __len__(self):
        return len(self.natural_data)


def get_dataloaders(natural_path='data/train', stylized_path='data/stylized', batch_size=64):
    """Returns the SimCLR-style training dataloader."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = StylizedPairDataset(natural_path, stylized_path, transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader
