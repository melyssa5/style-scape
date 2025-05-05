import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection_head = SimCLRProjectionHead(512, 512, 128)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.model = SimCLR(self.backbone)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.06)
    
    def prepare_data():
        transform = SimCLRTransform(input_size=32, gaussian_blur=0.0)



    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z