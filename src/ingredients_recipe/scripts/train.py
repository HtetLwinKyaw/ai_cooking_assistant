# src/ingredients_recipe/scripts/train.py

import os
import argparse
import torch
from torch.utils.data import DataLoader

from ingredients_recipe.data.dataset import IngredientDataset
from ingredients_recipe.models.model import IngredientModel
from ingredients_recipe.train.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ingredient prediction model")

    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--encoder", type=str, default="resnet50")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--unfreeze_after", type=int, default=-1)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


def save_checkpoint(model, optimizer, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    # ensure optimizer tensors are on correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return checkpoint["epoch"] + 1


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------
    # DATASETS (train / val)
    # ------------------------
    full_dataset = IngredientDataset(data_dir=args.data_dir)

    # simple 80/20 split
    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size

    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # safe for small datasets
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # ------------------------
    # MODEL
    # ------------------------
    model = IngredientModel(
        encoder_name=args.encoder,
        num_ingredients=full_dataset.num_ingredients,
        pretrained_encoder=True,
        freeze_encoder=args.freeze_encoder,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------
    # RESUME
    # ------------------------
    start_epoch = 1
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)

    # ------------------------
    # TRAINER
    # ------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # ------------------------
    # TRAIN LOOP
    # ------------------------
    for epoch in range(start_epoch, args.epochs + 1):

        if args.unfreeze_after > 0 and epoch == args.unfreeze_after:
            print("Unfreezing encoder for fine-tuning")
            model.unfreeze_encoder()

        train_loss = trainer.train_one_epoch(epoch)
        val_loss, metrics = trainer.validate(epoch)

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )

        # save epoch checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
        save_checkpoint(model, optimizer, epoch, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
