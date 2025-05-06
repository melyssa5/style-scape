import torch
import torch.nn as nn
import torchvision.models as models
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
from lightly.transforms.simclr_transform import SimCLRTransform


class SimCLRModel(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Backbone 
        resnet = models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  
        # Projection Head
        self.projection_head = SimCLRProjectionHead(512, 512, 128)
        # Linear Classification Head for evalutation
        self.linear_head = nn.Linear(512, 15)  # 15 classes
        self.loss_fn = NTXentLoss()  
        self.to(self.device)

    def forward(self, x, return_features=False, mode="contrastive"):
        x = self.backbone(x).squeeze()
        if mode == "contrastive":
            return self.projection_head(x)
        elif mode == "linear":
            return self.linear_head(x)
        elif return_features:
            return x
        return x

    def training_step(self, x1, x2):
        """Contrastive training step with NT-Xent loss."""
        self.train()
        z1 = self.forward(x1, mode="contrastive")
        z2 = self.forward(x2, mode="contrastive")
        loss = self.loss_fn(z1, z2)
        return loss

    def evaluate_linear(self, dataloader):
        """Evaluate frozen encoder + linear head on a labeled dataset to get
        predictions"""
        self.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                features = self.forward(x, return_features=True)
                preds = self.linear_head(features)
                predicted = torch.argmax(preds, dim=1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        return correct / total


    def freeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_dataloader(self, path, batch_size=64, eval_mode=False):
        transform = SimCLRTransform(input_size=32, gaussian_blur=0.0) is not eval_mode 
         

        
