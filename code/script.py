
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fix OpenMP
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce CUDA warnings

import torch
import torch.nn as nn
from transformers import Dinov2Model

class DinoV2(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", num_classes=15):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(model_name)  # Must be string!
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoV2(model_name="facebook/dinov2-base").to(device)  # Explicit model name
    print("✅ Model loaded successfully!")

if __name__ == "__main__":
    run_pipeline()