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
    epochs = 50
    lr = 1e-3
    freeze_backbone = True              # Linear probing by default

# ====================== MODEL ======================
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

# ====================== TRAINING ======================
def train():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoV2(config).to(device)
    train_loader, test_loader = get_dataloaders(config)
    params = model.classifier.parameters() if config.freeze_backbone else model.parameters()
    optimizer = torch.optim.Adam(params, lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0

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

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)

        acc = 100 * correct / total
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f} | Test Acc: {acc:.1f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "dinov2_best_model.pth")
            print(f"✅ Saved new best model (acc={acc:.2f}%)")

    torch.save(model.state_dict(), "dinov2_last_model.pth")
    print("📝 Training complete. Final model saved as 'dinov2_last_model.pth'")

# ======== testing =========
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = 100 * correct / total
    print(f"Evaluation Accuracy: {acc:.2f}%")
    return acc


# ========= LIME Samples ===============
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random

def run_lime_on_samples(model, dataloader, processor, device, output_dir="lime_outputs", num_samples=2):
    """
    Automatically runs LIME on selected correct and incorrect samples from the dataset.
    Saves visualizations in output_dir.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    correct_imgs, incorrect_imgs = [], []

    # First pass: collect predictions
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            preds = model(x).argmax(dim=1)
        
        for i in range(len(preds)):
            image_tensor = x[i].cpu()
            label = y[i].item()
            pred = preds[i].item()

            # Convert tensor back to image (undo processor)
            image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            entry = (image_np, label, pred)
            if label == pred:
                correct_imgs.append(entry)
            else:
                incorrect_imgs.append(entry)

    # Select random samples
    random.seed(42)
    selected_correct = random.sample(correct_imgs, min(num_samples, len(correct_imgs)))
    selected_incorrect = random.sample(incorrect_imgs, min(num_samples, len(incorrect_imgs)))

    all_selected = [("correct", img) for img in selected_correct] + [("incorrect", img) for img in selected_incorrect]

    # LIME explainer
    explainer = lime_image.LimeImageExplainer()

    def lime_predict(images_np):
        inputs = processor(images=[Image.fromarray(img).convert("RGB") for img in images_np], return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(inputs["pixel_values"])
            probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # Run LIME on each image
    for tag, (img_np, label, pred) in all_selected:
        explanation = explainer.explain_instance(
        img_np,
        lime_predict,
        labels=[pred],  # explicitly request explanation for pred
        hide_color=0,
        num_samples=1000
        )

        temp, mask = explanation.get_image_and_mask(
            label=pred,
            positive_only=True,
            num_features=5,
            hide_rest=False
        )


        # Save visualization
        fig, ax = plt.subplots()
        ax.imshow(mark_boundaries(temp, mask))
        ax.axis('off')
        ax.set_title(f"{tag} | pred: {pred} | label: {label}")
        fname = f"{output_dir}/lime_{tag}_pred{pred}_label{label}.png"
        plt.savefig(fname)
        plt.close()
        print(f"Saved: {fname}")

# ========= GradCAM Utils ===========
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import numpy as np
import random

def run_gradcam_on_image(model, processor, image_pil, label=None, device='cuda'):
    model.eval()
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    logits = model(inputs["pixel_values"])
    pred_class = logits.argmax(dim=1).item()
    target_class = label if label is not None else pred_class
    model.zero_grad()
    logits[:, target_class].backward()
    cam = model.get_cam().squeeze(0)
    num_patches = cam.shape[0] - 1
    size = int(num_patches**0.5)
    cam = cam[1:].reshape(size, size).cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam_resized = cv2.resize(cam, image_pil.size)
    return cam_resized, pred_class

def overlay_heatmap_on_image(image_pil, heatmap, save_path=None):
    heatmap_colored = cm.jet(heatmap)[..., :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    blended = Image.blend(image_pil.convert("RGB"), Image.fromarray(heatmap_colored), alpha=0.5)
    if save_path:
        blended.save(save_path)
    else:
        plt.imshow(blended)
        plt.axis('off')
        plt.title("Grad-CAM Overlay")
        plt.show()

def run_gradcam_on_samples(model, dataloader, processor, device, output_dir="gradcam_outputs", num_samples=2):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    correct_imgs, incorrect_imgs = [], []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            preds = model(x).argmax(dim=1)

        for i in range(len(preds)):
            image_tensor = x[i].cpu()
            label = y[i].item()
            pred = preds[i].item()
            image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            entry = (image_np, label, pred)
            if label == pred:
                correct_imgs.append(entry)
            else:
                incorrect_imgs.append(entry)

    selected_correct = random.sample(correct_imgs, min(num_samples, len(correct_imgs)))
    selected_incorrect = random.sample(incorrect_imgs, min(num_samples, len(incorrect_imgs)))
    all_selected = [("correct", img) for img in selected_correct] + [("incorrect", img) for img in selected_incorrect]

    for tag, (img_np, label, pred) in all_selected:
        image_pil = Image.fromarray(img_np)
        heatmap, _ = run_gradcam_on_image(model, processor, image_pil, label=label, device=device)
        fname = f"{output_dir}/gradcam_{tag}_pred{pred}_label{label}.png"
        overlay_heatmap_on_image(image_pil, heatmap, save_path=fname)
        print(f"Saved: {fname}")



if __name__ == "__main__":
    # train()
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoV2(config).to(device)
    model.load_state_dict(torch.load("dinov2_last_model.pth"))
    _, stylized_loader, natural_test_loader = get_dataloaders(config)
    processor = AutoImageProcessor.from_pretrained(config.model_name)
    print("\nEvaluating on Stylized Dataset:")
    evaluate_model(model, stylized_loader, device)
    print("\nEvaluating on Natural Test Dataset:")
    evaluate_model(model, natural_test_loader, device)
    run_lime_on_samples(model, stylized_loader, processor, device)
    run_gradcam_on_samples(model, stylized_loader, processor, device)
