# src/ingredients_recipe/scripts/train_dish_classifier.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


# -------------------------
# CONFIG
# -------------------------
DATA_ROOT = "data/dish"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")

NUM_CLASSES = 100
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

CHECKPOINT_DIR = "checkpoints/dish"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -------------------------
# MAIN
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # DATA
    # -------------------------
    train_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # -------------------------
    # MODEL
    # -------------------------
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    # replace classifier
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

    # -------------------------
    # TRAIN LOOP
    # -------------------------
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                preds = logits.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # -------------------------
        # CHECKPOINT
        # -------------------------
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": train_ds.class_to_idx,
                },
                ckpt_path,
            )
            print(f"✅ Saved best model (acc={best_acc:.4f})")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
