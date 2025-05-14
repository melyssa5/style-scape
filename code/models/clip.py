import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from ..train import train_epoch, evaluate_accuracy

class CLIPClassifier(nn.Module):
    def __init__(self, num_classes=15, freeze_backbone=True):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        if freeze_backbone:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, pixel_values):
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.model.vision_model.parameters())):
            image_embeds = self.model.get_image_features(pixel_values=pixel_values)
        return self.classifier(image_embeds)

    @property
    def transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

def collate_fn(batch, processor):
    images, labels = zip(*batch)
    pixel_values = processor(images=list(images), return_tensors="pt", do_rescale=False)["pixel_values"]
    return pixel_values, torch.tensor(labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "../../data"
    batch_size = 32
    num_epochs = 50
    num_classes = 15

    model = CLIPClassifier(num_classes=num_classes).to(device)
    tf = model.transform
    processor = model.processor

    train_data = ImageFolder(f"{data_dir}/train", transform=tf)
    natural_test_data = ImageFolder(f"{data_dir}/test", transform=tf)
    stylized_test_data = ImageFolder(f"{data_dir}/test", transform=tf)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch, processor))
    natural_test_loader = DataLoader(natural_test_data, batch_size=batch_size,
                             collate_fn=lambda batch: collate_fn(batch, processor))
    stylized_test_loader = DataLoader(stylized_test_data, batch_size=batch_size,
                             collate_fn=lambda batch: collate_fn(batch, processor))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_loss = float('inf')
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1} | Loss: {loss:.4f}")
        if loss < best_loss:
            torch.save(model.state_dict(), "best_alexnet.pth")
            best_loss = loss
            print(f"New best model saved (Loss: {best_loss:.4f})")


    accuracy = evaluate_accuracy(model, natural_test_loader, device)
    print(f"\nNatural Test Accuracy: {accuracy:.2%}")
    accuracy = evaluate_accuracy(model, stylized_test_loader, device)
    print(f"\nStylized Test Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
