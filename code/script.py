import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import Dinov2Model, AutoImageProcessor
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import pandas as pd
from models.dinoV2 import DinoV2
from data import get_dataloaders
from train import evaluate, train_epoch

# --- Config ---
class Config:
    data_dir = "data"  # Update this path in Colab
    model_name = "facebook/dinov2-small"
    num_classes = 15
    batch_size = 32
    epochs = 5
    lr = 1e-3
    freeze_backbone = True  # Linear probing by default

def run_pipeline():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize
    model = DinoV2(config).to(device)
    train_loader, test_loader = get_dataloaders(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Train Loss: {train_loss:.4f}")
        evaluate(model, test_loader, device)
    
    # Save model
    torch.save(model.state_dict(), "dino_model.pth")
    print("\nModel saved to 'dino_model.pth'")

if __name__ == "__main__":
    run_pipeline()