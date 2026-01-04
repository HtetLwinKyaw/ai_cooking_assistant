# src/ingredients_recipe/scripts/infer.py

import argparse
import torch
from PIL import Image
from torchvision import transforms

from ingredients_recipe.data.dataset import IngredientDataset
from ingredients_recipe.models.model import IngredientModel


def parse_args():
    parser = argparse.ArgumentParser(description="Ingredient inference from image")

    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--encoder", type=str, default="resnet50")
    parser.add_argument("--threshold", type=float, default=0.5)

    return parser.parse_args()


def load_image(image_path: str):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # [1, 3, 224, 224]


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset ONLY to get ingredient vocabulary
    dataset = IngredientDataset(data_dir=args.data_dir)
    ingredient_vocab = dataset.ingredient_vocab

    # Build model
    model = IngredientModel(
        encoder_name=args.encoder,
        num_ingredients=len(ingredient_vocab),
        pretrained_encoder=False,
        freeze_encoder=True,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # support both full checkpoint and state_dict-only
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Load image
    image = load_image(args.image).to(device)

    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits).squeeze(0)

    print("\nPredicted ingredients:")
    for ing, p in zip(ingredient_vocab, probs):
        if p.item() >= args.threshold:
            print(f"  {ing:20s}  {p.item():.3f}")


if __name__ == "__main__":
    main()
