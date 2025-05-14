import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd
from bagnet import bagnet33 # type: ignore
from ..train import train_epoch, evaluate_accuracy

class BagNetClassifier(nn.Module):
    def __init__(self, num_classes=15, pretrained=True):
        super().__init__()
        self.backbone = bagnet33(pretrained=pretrained)
        self.backbone.fc = nn.Identity()  # Remove original classifier
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.backbone(x)  
        pooled = features.mean([2, 3])  # Global average pooling
        return self.classifier(pooled)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "../../data"
    batch_size = 32
    num_epochs = 50
    num_classes = 15
    
    # Model Specific transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_data = ImageFolder(f"{data_dir}/train", transform=transform)
    natural_test_data = ImageFolder(f"{data_dir}/test", transform=transform)
    stylized_test_data = ImageFolder(f"{data_dir}/stylized", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    natural_test_loader = DataLoader(natural_test_data, batch_size=batch_size)
    stylized_test_loader = DataLoader(stylized_test_data, batch_size=batch_size)

    model = BagNetClassifier(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float('inf')
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1} | Loss: {loss:.4f}")
        if loss < best_loss:
            torch.save(model.state_dict(), "best_bagnet.pth")
            best_loss = loss
            print(f"New best model saved (Loss: {best_loss:.4f})")

    accuracy = evaluate_accuracy(model, natural_test_loader, device)
    print(f"\nNatural Test Accuracy: {accuracy:.2%}")

    accuracy = evaluate_accuracy(model, stylized_test_loader, device)
    print(f"\nStylized Test Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()