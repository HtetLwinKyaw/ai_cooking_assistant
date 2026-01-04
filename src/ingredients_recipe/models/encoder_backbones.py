# src/ingredients_recipe/models/encoder_backbones.py

import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    """
    Generic image encoder using ImageNet pretrained backbones.
    Supports: resnet50, efficientnet_b0, mobilenet_v3_small
    """

    def __init__(self, name: str = "resnet50", pretrained: bool = True, trainable: bool = False):
        super().__init__()

        self.name = name.lower()
        self.pretrained = pretrained
        self.trainable = trainable

        if self.name == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = backbone.fc.in_features
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        elif self.name == "efficientnet_b0":
            backbone = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.encoder = backbone

        elif self.name == "mobilenet_v3_small":
            backbone = models.mobilenet_v3_small(pretrained=pretrained)
            self.feature_dim = backbone.classifier[3].in_features
            backbone.classifier = nn.Identity()
            self.encoder = backbone

        else:
            raise ValueError(f"Unsupported encoder: {name}")

        # Freeze encoder by default (transfer learning)
        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x -> [B, 3, H, W]
        Output: features -> [B, feature_dim]
        """
        feats = self.encoder(x)

        # ResNet outputs [B, C, 1, 1]
        if feats.dim() == 4:
            feats = feats.flatten(1)

        return feats

    def unfreeze(self):
        """Unfreeze encoder for fine-tuning"""
        for p in self.encoder.parameters():
            p.requires_grad = True
