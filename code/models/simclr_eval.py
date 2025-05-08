import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

from pytorch_lightning import seed_everything
from pathlib import Path
from simclr import SimCLR

# Load trained encoder
CHECKPOINT_PATH = 'lightning_logs/version_1/checkpoints'
ckpt_file = [f for f in Path(CHECKPOINT_PATH).rglob('*.ckpt')][0]
simclr_model = SimCLR.load_from_checkpoint(str(ckpt_file), strict=False)
encoder = simclr_model.encoder
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

# Dataset paths
train_dir = "data/train"
test_dir = "data/test"
stylized_dir = "data/stylized"

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Datasets
train_dataset = ImageFolder(train_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)
stylized_dataset = ImageFolder(stylized_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
stylized_loader = DataLoader(stylized_dataset, batch_size=64, shuffle=False, num_workers=2)

# Classifier head
classifier = nn.Sequential(
    encoder,
    nn.Linear(128, 15)  # 15 scene categories
).to("cuda" if torch.cuda.is_available() else "cpu")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier[1].parameters(), lr=1e-3)

# TensorBoard
writer = SummaryWriter(log_dir="logs/simclr_linear_eval")

# Training loop
epochs = 50
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
    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Train Accuracy = {acc:.4f}")
    writer.add_scalar("Train/Loss", avg_loss, epoch)
    writer.add_scalar("Train/Accuracy", acc, epoch)

# Final test accuracy on natural images
classifier.eval()
correct = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = classifier(imgs)
        correct += (outputs.argmax(1) == labels).sum().item()
test_acc = correct / len(test_dataset)
print(f"\nFinal Test Accuracy (Natural): {test_acc:.4f}")
writer.add_scalar("Test/Natural_Accuracy", test_acc, epochs)

# Final test accuracy on stylized images
classifier.eval()
correct_stylized = 0
with torch.no_grad():
    for imgs, labels in stylized_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = classifier(imgs)
        correct_stylized += (outputs.argmax(1) == labels).sum().item()

stylized_acc = correct_stylized / len(stylized_dataset)
print(f"\n Final Stylized Test Accuracy: {stylized_acc:.4f}")
writer.add_scalar("Test/Accuracy_Stylized", stylized_acc, epochs)

writer.close()

#####GRAD CAM #####
import os
from torchvision.utils import save_image
os.makedirs("gradcam_outputs", exist_ok=True)

print("\nGenerating Grad-CAM visualizations...")



# Grad-CAM setup
cam_extractor = GradCAM(classifier, target_layer="0.layer4")
output_dir = "gradcam_outputs"
os.makedirs(output_dir, exist_ok=True)

# Pick a few stylized samples to visualize
sample_imgs, sample_labels = next(iter(test_loader))
sample_imgs = sample_imgs.to(device)

# Get predictions and activation maps
# Enable gradients for encoder
for param in classifier[0].parameters():
    param.requires_grad = True

classifier.train()  # Important: CAM needs model in train mode (no torch.no_grad)


for i in range(min(5, len(sample_imgs))):  # Visualize 5 samples max
    img = sample_imgs[i].unsqueeze(0)
    output = classifier(img)
    class_idx = output.argmax(dim=1).item()

    # Get Grad-CAM activation
    activation_map = cam_extractor(class_idx, output)

    # Prepare original image
    original_img = to_pil_image(sample_imgs[i].cpu())

    # Normalize activation map to [0,1]
    act = activation_map[0].squeeze(0).cpu()
    act = (act - act.min()) / (act.max() - act.min() + 1e-5)

    # Overlay Grad-CAM heatmap
    result = overlay_mask(original_img, to_pil_image(act, mode='F'), alpha=0.5)

    # Save the visualization
    result.save(f"{output_dir}/sample_{i}_class_{class_idx}.png")
    print(f"Saved: sample_{i}_class_{class_idx}.png")

