import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from torchvision import models

# MODEL PARAMETERS
NUM_EPOCHS = 90
MOMENTUM = 0.9
LEARNING_RATE = 0.01
LR_DECAY = 0.0005

class AlexNet(nn.Module):
    def __init__(self, num_classes=15, pretrained=True):
        super().__init__()
        self.model = models.alexnet(pretrained=pretrained)
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        
        # ImageNet normalization + augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Validation/testing transform
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        return self.model(x)

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss / len(train_loader)

def test(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

def main():
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    epochs = 20
    
    # Initialize
    model = AlexNet(num_classes=15).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )
    criterion = nn.CrossEntropyLoss()

    # Data
    train_data = datasets.ImageFolder(
        "data/train",
        transform=model.train_transform
    )
    test_data = datasets.ImageFolder(
        "data/stylized",
        transform=model.test_transform
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Training phase
    best_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
        
        # Save best model based on training loss
        if train_loss < best_loss:
            torch.save(model.state_dict(), "best_alexnet.pth")
            best_loss = train_loss
            print(f"New best model saved (Loss: {best_loss:.4f})")

    # Testing phase (after all training is complete)
    print("\nTraining complete. Starting testing...")
    model.load_state_dict(torch.load("best_alexnet.pth"))  # Load best model
    test_acc = test(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {100*test_acc:.2f}%")

if __name__ == "__main__":
    main()

# Sources Used
# https://github.com/dansuh17/alexnet-pytorch 
# https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html 