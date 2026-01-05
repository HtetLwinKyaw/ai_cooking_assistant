# scripts/prepare_uecfood_dish.py

import shutil
import random
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
UEC_ROOT = Path(
    r"C:\FORSAKEN\Projects\ai_cooking_assistant\src\ingredients_recipe\data\UECFOOD100"
)

OUTPUT_ROOT = Path("data/dish")
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

# -------------------------
# MAIN
# -------------------------
def main():
    assert UEC_ROOT.exists(), f"UECFood root not found: {UEC_ROOT}"

    train_root = OUTPUT_ROOT / "train"
    val_root = OUTPUT_ROOT / "val"

    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted(
        [d for d in UEC_ROOT.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name),
    )

    total_images = 0

    for class_dir in class_dirs:
        images = list(class_dir.glob("*.jpg"))
        if not images:
            continue

        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)

        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        (train_root / class_dir.name).mkdir(exist_ok=True)
        (val_root / class_dir.name).mkdir(exist_ok=True)

        for img in train_imgs:
            shutil.copy(img, train_root / class_dir.name / img.name)

        for img in val_imgs:
            shutil.copy(img, val_root / class_dir.name / img.name)

        total_images += len(images)

        print(
            f"Class {class_dir.name}: "
            f"{len(train_imgs)} train / {len(val_imgs)} val"
        )

    print("\n✅ Dish dataset prepared")
    print(f"Total images: {total_images}")
    print(f"Train path: {train_root.resolve()}")
    print(f"Val path: {val_root.resolve()}")


if __name__ == "__main__":
    main()
