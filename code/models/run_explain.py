import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
import numpy as np
from PIL import Image
from explain import (
    get_default_transform,
    run_lime_interpreter_torch,
    run_gradcam_interpreter_torch
)

def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    transformed = transform(image)
    image_np = np.array(image)
    return transformed.unsqueeze(0), image_np

def run_explain_on_image(model, weight_path, target_layer, image_path, class_names, save_prefix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and weights
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    # Load image and preprocess
    transform = get_default_transform()
    input_tensor, image_np = load_and_preprocess_image(image_path, transform)
    input_tensor = input_tensor.to(device)

    # Run prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_label = output.argmax(dim=1).item()

    print("⚙️  Running explainers...")
    prefix = f"{save_prefix}_{os.path.basename(image_path).split('.')[0]}"

    run_lime_interpreter_torch(model, image_np, class_names, device,
                               pred_label=pred_label, filename_prefix=prefix,
                               save_dir=f"lime_{save_prefix}")

    run_gradcam_interpreter_torch(model, image_np, class_names, device,
                                  pred_label=pred_label, filename_prefix=prefix,
                                  save_dir=f"gradcam_{save_prefix}",
                                  target_layer=target_layer)




