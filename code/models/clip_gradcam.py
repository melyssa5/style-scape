import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import os


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
    return pixel_values, torch.tensor(labels), images

def visualize_gradcam(model, processor, data_loader, device, save_dir="clip_gradcam_outputs"):
    os.makedirs(save_dir, exist_ok=True)

    cam_extractor = GradCAM(
        model=model,
        target_layer="model.vision_model.encoder.layers.11.layer_norm"
    )

    model.eval()
    for inputs, labels, raw_imgs in data_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        pred_classes = outputs.argmax(dim=1)

        for i in range(min(5, len(inputs))):
            img_tensor = inputs[i].unsqueeze(0)
            output = model(img_tensor)
            class_idx = output.argmax().item()

            # Get Grad-CAM
            activation_map = cam_extractor(class_idx, output)[0]
            act = activation_map.squeeze().cpu()
            act = (act - act.min()) / (act.max() - act.min() + 1e-5)

            original_img = raw_imgs[i]
            heatmap = overlay_mask(to_pil_image(original_img), to_pil_image(act, mode='F'), alpha=0.5)
            heatmap.save(os.path.join(save_dir, f"clip_cam_{i}_class_{class_idx}.png"))
        break  # Just visualize one batch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPClassifier().to(device)
    processor = model.processor
    tf = model.transform

    test_dataset = ImageFolder("../../data/test", transform=tf)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             collate_fn=lambda batch: collate_fn(batch, processor))

    model.load_state_dict(torch.load("best_alexnet.pth", map_location=device))
    visualize_gradcam(model, processor, test_loader, device)

if __name__ == "__main__":
    main()
