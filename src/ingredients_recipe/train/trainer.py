# src/ingredients_recipe/train/trainer.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ingredients_recipe.utils.metrics import multilabel_metrics


class Trainer:
    """
    Trainer for ingredient prediction with validation + metrics.
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        optimizer,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.best_f1 = 0.0

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch in pbar:
            images = batch["image"].to(self.device)
            targets = batch["ingredients"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch: int):
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

        for batch in pbar:
            images = batch["image"].to(self.device)
            targets = batch["ingredients"].to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

        avg_loss = total_loss / len(self.val_loader)

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)

        metrics = multilabel_metrics(logits, targets)

        return avg_loss, metrics

    def fit(self, epochs: int):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch)

            if self.val_loader is not None:
                val_loss, metrics = self.validate(epoch)

                print(
                    f"Epoch {epoch} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"F1: {metrics['f1']:.4f}"
                )

                # save best model
                if metrics["f1"] > self.best_f1:
                    self.best_f1 = metrics["f1"]
                    path = f"{self.checkpoint_dir}/best.pt"
                    torch.save(self.model.state_dict(), path)
                    print(f"Saved BEST model (F1={self.best_f1:.4f})")

            else:
                print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")
