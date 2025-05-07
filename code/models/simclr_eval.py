import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from pytorch_lightning import seed_everything
from simclr import SimCLR


# Load trained encoder
CHECKPOINT_PATH = 'lightning_logs/version_0/checkpoints'
ckpt_file = [f for f in Path(CHECKPOINT_PATH).rglob('*.ckpt')][0]
simclr_model = SimCLR.load_from_checkpoint(str(ckpt_file), strict=False)
encoder = simclr_model.encoder
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

# Dataset paths
train_dir = "style-scape/data/train"
test_dir = "style-scape/data/test"

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Datasets
train_dataset = ImageFolder(train_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Classifier head
classifier = nn.Sequential(
    encoder,
    nn.Linear(512, 15)  # 15 scene categories
).to("cuda" if torch.cuda.is_available() else "cpu")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier[1].parameters(), lr=1e-3)

# Training loop
epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
for epoch in range(epochs):
    classifier.train()
    total_loss, total_correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = classifier(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()

    acc = total_correct / len(train_dataset)
    print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Train Accuracy = {acc:.4f}")

# Final test accuracy
classifier.eval()
correct = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = classifier(imgs)
        correct += (outputs.argmax(1) == labels).sum().item()

test_acc = correct / len(test_dataset)
print(f"\n Final Test Accuracy: {test_acc:.4f}")
