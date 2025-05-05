from matplotlib import transforms
import torch
import torch.nn as nn
from torchvision import models
import tensorflow as tf


# MODEL PARAMETERS
NUM_EPOCHS = 90
MOMENTUM = 0.9
LEARNING_RATE = 0.01
LR_DECAY = 0.0005

class AlexNet(nn.Module):
    def __init__(self, pretrained=True, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load AlexNet
        self.model = models.alexnet(pretrained=pretrained)

        # Replace the final classifier layer for custom task
        self.model.classifier[6] = nn.Linear(4096, 15)
        self.model = self.model.to(self.device)

        # ImageNet-style preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to expected input
            transforms.ToTensor(),         # Convert PIL to Tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, weight_decay=LR_DECAY)

    @staticmethod
    def loss_fn(labels, predictions):
        cce = tf.keras.losses.SparseCategoricalCrossentropy()
        return cce(labels, predictions)


# Sources Used
# https://github.com/dansuh17/alexnet-pytorch 
# https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html 