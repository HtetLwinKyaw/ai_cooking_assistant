# src/ingredients_recipe/utils/metrics.py

import torch


def multilabel_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
):
    """
    Compute micro precision / recall / F1 for multi-label classification.

    logits:  [B, C] raw logits
    targets: [B, C] binary ground truth (0/1)
    """

    # convert logits -> binary predictions
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    # flatten
    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }
