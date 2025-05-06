# train.py
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor, Dinov2Model
from tqdm import tqdm
import os

# ====================== CONFIG ======================
class Config:
    data_dir = "../data"                   # Path to your data folder
    model_name = "facebook/dinov2-small" # DINOv2 variant
    num_classes = 15                    # Update based on your dataset
    batch_size = 32
    epochs = 10
    lr = 1e-3
    freeze_backbone = True              # Linear probing by default

# ====================== MODEL ======================
class DinoV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(config.model_name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, config.num_classes)
        
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        return self.classifier(outputs.last_hidden_state[:, 0])

# ====================== DATA ======================
def get_dataloaders(config):
    # DINOv2-specific preprocessing (no rescaling!)
    processor = AutoImageProcessor.from_pretrained(config.model_name)
    
    # Only resize + convert to tensor (DINOv2 handles normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: x.convert('RGB'))
    ])
    
    # Create datasets
    train_set = ImageFolder(f"{config.data_dir}/train", transform=transform)
    test_set = ImageFolder(f"{config.data_dir}/stylized", transform=transform)
    
    # Custom collate function
    def collate_fn(batch):
        images, labels = zip(*batch)
        inputs = processor(
            images=list(images),
            return_tensors="pt",
            do_rescale=False  # Maintain [0,255] range
        )
        return inputs['pixel_values'], torch.tensor(labels)
    
    return (
        DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn),
        DataLoader(test_set, batch_size=config.batch_size, collate_fn=collate_fn)
    )

# ====================== TRAINING ======================
def train():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize
    model = DinoV2(config).to(device)
    train_loader, test_loader = get_dataloaders(config)
    
    # Optimizer (only train classifier if backbone frozen)
    params = model.classifier.parameters() if config.freeze_backbone else model.parameters()
    optimizer = torch.optim.Adam(params, lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Evaluation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Test Acc: {100*correct/total:.1f}%")

    # Save model
    torch.save(model.state_dict(), "dino_style_transfer.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()