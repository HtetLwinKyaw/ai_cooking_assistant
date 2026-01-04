# src/ingredients_recipe/models/model.py

import torch
import torch.nn as nn

from ingredients_recipe.models.encoder_backbones import ImageEncoder
from ingredients_recipe.models.heads import IngredientHead


class IngredientModel(nn.Module):
    """
    Full model for image -> ingredient prediction using transfer learning.

    Architecture:
      ImageEncoder (pretrained CNN, frozen by default)
        -> IngredientHead (trainable MLP)
    """

    def __init__(
        self,
        encoder_name: str,
        num_ingredients: int,
        pretrained_encoder: bool = True,
        freeze_encoder: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Image encoder
        self.encoder = ImageEncoder(
            name=encoder_name,
            pretrained=pretrained_encoder,
            trainable=not freeze_encoder,
        )

        # Ingredient prediction head
        self.ingredient_head = IngredientHead(
            feature_dim=self.encoder.feature_dim,
            num_ingredients=num_ingredients,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W]
        returns: ingredient logits [B, num_ingredients]
        """
        features = self.encoder(images)
        logits = self.ingredient_head(features)
        return logits

    def unfreeze_encoder(self):
        """Enable fine-tuning of the encoder"""
        self.encoder.unfreeze()
