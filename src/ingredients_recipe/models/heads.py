# src/ingredients_recipe/models/heads.py

import torch
import torch.nn as nn


class IngredientHead(nn.Module):
    """
    Multi-label ingredient classifier.
    Input: image features [B, feature_dim]
    Output: ingredient logits [B, num_ingredients]
    """

    def __init__(
        self,
        feature_dim: int,
        num_ingredients: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
    nn.Linear(feature_dim, hidden_dim),
    nn.ReLU(inplace=True),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, num_ingredients),
)


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: [B, feature_dim]
        returns: logits [B, num_ingredients]
        """
        return self.classifier(features)
