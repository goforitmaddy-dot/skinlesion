import torch
import torch.nn as nn
import timm

class SkinLesionModel(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits