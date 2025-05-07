import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from glob import glob
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Custom Dataset
class StylizedSimCLRDataset(data.Dataset):
    def __init__(self, train_dir, stylized_dir, transform=None):
        self.train_dir = train_dir
        self.stylized_dir = stylized_dir
        self.transform = transform
        self.train_images = sorted(glob(os.path.join(train_dir, '*', '*.jpg')))

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        orig_path = self.train_images[idx]
        label = orig_path.split(os.sep)[-2]
        filename = os.path.basename(orig_path).split('.')[0]  # no extension
        style_glob = os.path.join(self.stylized_dir, label, filename + '-stylized-*.jpg')
        style_paths = sorted(glob(style_glob))

        image = Image.open(orig_path).convert('RGB')
        stylized_image = Image.open(style_paths[0]).convert('RGB')  # take first stylized variant

        if self.transform:
            image = self.transform(image)
            stylized_image = self.transform(stylized_image)

        return (image, stylized_image)

# SimCLR Model
class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim=128, lr=1e-3, temperature=0.5, weight_decay=1e-4, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = models.resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(512, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]

    def info_nce_loss(self, z):
        z = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z, z.T) / self.hparams.temperature
        batch_size = z.size(0) // 2
        labels = torch.arange(batch_size).to(self.device)
        labels = torch.cat([labels, labels], dim=0)
        mask = torch.eye(len(z), dtype=torch.bool).to(self.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def forward(self, x):
        return self.projector(self.encoder(x))

    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        z1 = self(x1)
        z2 = self(x2)
        loss = self.info_nce_loss(torch.cat([z1, z2], dim=0))
        self.log("train_loss", loss)
        return loss

# Set paths and transformations
train_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = StylizedSimCLRDataset(
    train_dir='data/train',
    stylized_dir='data/stylized',
    transform=train_transform
)

train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4)

# Training
model = SimCLR()
trainer = pl.Trainer(
    max_epochs=100,
    devices=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    callbacks=[
        ModelCheckpoint(save_top_k=1, monitor="train_loss"),
        LearningRateMonitor(logging_interval='epoch')
    ]
)

trainer.fit(model, train_loader)
