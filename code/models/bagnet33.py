# bagnet33.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

class BagNet33(nn.Module):
    def __init__(self, num_classes=15, pretrained=True):
        super().__init__()
        # Load pretrained BagNet-33
        self.model = torch.hub.load('facebookresearch/bagnets', 'bagnet33', pretrained=pretrained)
        
        # Replace classifier
        self.model.fc = nn.Linear(512, num_classes)
        
        # BagNet-specific preprocessing (no resizing!)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        return self.model(x)

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(test_loader.dataset)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize
    model = BagNet33(num_classes=15).to(device)
    optimizer = torch.optim.SGD([
        {'params': model.model.fc.parameters(), 'lr': 0.01},  # Higher LR for head
        {'params': model.model.parameters(), 'lr': 0.001}     # Lower LR for backbone
    ], momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Data (no resizing - BagNet expects original sizes)
    train_data = datasets.ImageFolder("data/train", transform=model.transform)
    test_data = datasets.ImageFolder("data/stylized", transform=model.transform)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    # Training phase
    for epoch in range(10):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")
    
    # Testing phase (after full training)
    test_acc = test(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {test_acc:.2%}")

if __name__ == "__main__":
    main()