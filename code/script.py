import torch
import torch.nn as nn

from transformers import AutoImageProcessor, Dinov2Model
from data import get_dataloaders
from train import evaluate, train_epoch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fix OpenMP warning

class DinoV2(nn.Module):
    def __init__(self, model_name='facebook/dinov2-small', num_classes=15, freeze_backbone=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained components
        self.backbone = Dinov2Model.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Classifier head
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)
        
        # Freeze backbone by default (recommended for linear probing)
        if freeze_backbone:
            self.freeze_encoder()
            
        self.to(self.device)

    def forward(self, x):
        """Forward pass with CLS token (better than mean pooling for DINOv2)"""
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]  # Use CLS token instead of mean pooling
        return self.classifier(cls_token)

    def process_input(self, images):
        """Process images using DINOv2's processor"""
        return self.processor(images=images, return_tensors="pt", do_rescale=False)["pixel_values"]

    def freeze_encoder(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()  # Important for dropout/batchnorm

    def unfreeze_encoder(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()

    def get_trainable_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
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