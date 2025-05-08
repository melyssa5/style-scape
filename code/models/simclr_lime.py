import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from simclr import SimCLR

from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os

# Load trained encoder
CHECKPOINT_PATH = 'lightning_logs/version_1/checkpoints'
ckpt_file = [f for f in Path(CHECKPOINT_PATH).rglob('*.ckpt')][0]
simclr_model = SimCLR.load_from_checkpoint(str(ckpt_file), strict=False)
encoder = simclr_model.encoder
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

# Define classifier
classifier = nn.Sequential(
    encoder,
    nn.Linear(128, 15)  # 15 scene categories
).to("cuda" if torch.cuda.is_available() else "cpu")

# Load trained weights for linear head if applicable
# (Assumes you trained classifier separately and saved weights)
# classifier.load_state_dict(torch.load('path_to_linear_head.pth'))

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Prediction function for LIME
from torchvision.transforms.functional import to_tensor

def predict_fn(images_np):
    classifier.eval()
    batch = torch.stack([to_tensor(Image.fromarray(img)).to(device) for img in images_np])
    with torch.no_grad():
        logits = classifier(batch)
        probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.cpu().numpy()

# Pick a sample image
sample_path = "style-scape/data/stylized/Bedroom/image_0002-stylized-expressionism-style.jpg"  # <-- Replace with actual path
img = Image.open(sample_path).convert("RGB")
img_np = np.array(img)

# LIME setup
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(img_np,
                                         predict_fn,
                                         top_labels=1,
                                         hide_color=0,
                                         num_samples=1000)

# Visualize
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], 
    positive_only=True, 
    num_features=5, 
    hide_rest=False
)

plt.figure(figsize=(6, 6))
plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.title("LIME Explanation")
plt.axis("off")
plt.tight_layout()
plt.savefig("lime_output.png")
plt.show()
