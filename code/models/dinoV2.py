# train.py
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor, Dinov2Model
from ..train import train_epoch, evaluate_accuracy

class DinoV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(config.model_name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, config.num_classes)

        self.gradients = None
        self.activations = None

        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Hook into the last transformer block
        target_block = self.backbone.encoder.layer[-1]  # last transformer block

        def save_activation(module, input, output):
            self.activations = output[0].detach()  # output is a tuple

        def save_gradient(grad):
            self.gradients = grad

        target_block.register_forward_hook(save_activation)
        target_block.register_full_backward_hook(lambda m, g_in, g_out: save_gradient(g_out[0]))

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)

    def get_cam(self):
        weights = self.gradients.mean(dim=1, keepdim=True)
        cam = (weights * self.activations).sum(dim=-1)
        return cam

# ====================== DATA ======================
def get_dataloaders(config):
    processor = AutoImageProcessor.from_pretrained(config.model_name)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: x.convert('RGB'))
    ])

    train_set = ImageFolder(f"{config.data_dir}/train", transform=transform)
    test_set = ImageFolder(f"{config.data_dir}/stylized", transform=transform)
    natural_test_set = ImageFolder(f"{config.data_dir}/test", transform=transform)

    def collate_fn(batch):
        images, labels = zip(*batch)
        inputs = processor(
            images=list(images),
            return_tensors="pt",
            do_rescale=False
        )
        return inputs['pixel_values'], torch.tensor(labels)

    return (
        DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn),
        DataLoader(test_set, batch_size=config.batch_size, collate_fn=collate_fn),
        DataLoader(natural_test_set, batch_size=config.batch_size, collate_fn=collate_fn)
    )

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoV2(config).to(device)
    train_loader, test_loader, _ = get_dataloaders(config)
    params = model.classifier.parameters()
    optimizer = torch.optim.Adam(params, lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(50):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        acc = evaluate_accuracy(model, test_loader, device)
        print(f"Epoch {epoch+1:02d} | Train Loss: {loss:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "dinov2_best_model.pth")
            print(f"✅ Saved new best model (acc={acc:.2f}%)")

    torch.save(model.state_dict(), "dinov2_last_model.pth")
    

if __name__ == "__main__":
    data_dir = "../data"                   
    model_name = "facebook/dinov2-small" 
    num_classes = 15                    
    batch_size = 32
    epochs = 50
    lr = 1e-3
    freeze_backbone = True 
    # train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoV2().to(device)
    model.load_state_dict(torch.load("dinov2_last_model.pth"))
    _, stylized_loader, natural_test_loader = get_dataloaders()
    processor = AutoImageProcessor.from_pretrained(model_name)
    print("\nEvaluating on Stylized Dataset:")
    evaluate_accuracy(model, stylized_loader, device)
    print("\nEvaluating on Natural Test Dataset:")
    evaluate_accuracy(model, natural_test_loader, device)

