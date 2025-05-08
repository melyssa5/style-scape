import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


class ExpandedStylizedPairDataset(Dataset):
    def __init__(self, natural_root, stylized_root, transform=None):
        self.natural_data = ImageFolder(natural_root)
        self.stylized_data = ImageFolder(stylized_root)
        self.transform = transform
        assert len(self.stylized_data) == len(self.natural_data) * 3, \
            f"Expected 3 stylized per natural. Got {len(self.stylized_data)} stylized and {len(self.natural_data)} natural."

        self.total_pairs = len(self.natural_data) * 3

    def __getitem__(self, idx):
        nat_idx = idx // 3
        style_variant = idx % 3
        img1, _ = self.natural_data[nat_idx]
        stylized_idx = nat_idx * 3 + style_variant
        img2, _ = self.stylized_data[stylized_idx]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), 0  # dummy label

    def __len__(self):
        return self.total_pairs


def get_dataloaders(natural_path='/content/style-scape/data/train', stylized_path='/content/style-scape/data/stylized', batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = ExpandedStylizedPairDataset(natural_path, stylized_path, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader


class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim=128, lr=1e-3, temperature=0.5, weight_decay=1e-6):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.fc = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )

    def forward(self, x):
        return self.encoder(x)

    def info_nce_loss(self, features):
        batch_size = features.shape[0] // 2
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(self.device)

        sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        sim_matrix = sim_matrix / self.hparams.temperature

        sim_matrix = sim_matrix - torch.eye(sim_matrix.size(0), device=self.device) * 1e9
        positives = sim_matrix * labels
        negatives = sim_matrix * (1 - labels)

        nominator = torch.exp(positives).sum(dim=1)
        denominator = torch.exp(sim_matrix).sum(dim=1)
        loss = -torch.log(nominator / denominator)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        (x1, x2), _ = batch
        x = torch.cat([x1, x2], dim=0)
        z = self.encoder(x)
        loss = self.info_nce_loss(z)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer


def train_simclr():
    train_loader = get_dataloaders()
    model = SimCLR()
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[ModelCheckpoint(monitor='train_loss'), LearningRateMonitor(logging_interval='epoch')]
    )
    trainer.fit(model, train_loader)
    return model


if __name__ == '__main__':
    print("Training SimCLR with stylized dataset...")
    model = train_simclr()
    print("Training complete.")
