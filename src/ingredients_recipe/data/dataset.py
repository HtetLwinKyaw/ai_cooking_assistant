# src/ingredients_recipe/data/dataset.py

import os
import csv
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class IngredientDataset(Dataset):
    """
    Dataset for image -> multi-label ingredient prediction.

    Expected structure:
    data_dir/
      images/
        img1.jpg
        img2.jpg
      metadata.csv

    metadata.csv columns:
      image, ingredients

    ingredients column example:
      "salt,pepper,olive oil"
    """

    def __init__(
        self,
        data_dir: str,
        img_size: int = 224,
        ingredient_vocab: List[str] | None = None,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "images")
        self.csv_path = os.path.join(data_dir, "metadata.csv")

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"metadata.csv not found at {self.csv_path}")

        # load CSV
        self.samples = []
        all_ingredients = []

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = row["image"].strip()
                ingredients = [
                    x.strip().lower()
                    for x in row["ingredients"].split(",")
                    if x.strip()
                ]
                self.samples.append(
                    {
                        "image": image_name,
                        "ingredients": ingredients,
                    }
                )
                all_ingredients.extend(ingredients)

        # build ingredient vocabulary if not provided
        if ingredient_vocab is None:
            unique_ingredients = sorted(set(all_ingredients))
            self.ingredient_vocab = unique_ingredients
        else:
            self.ingredient_vocab = ingredient_vocab

        self.ingredient_to_idx = {
            ing: idx for idx, ing in enumerate(self.ingredient_vocab)
        }

        self.num_ingredients = len(self.ingredient_vocab)

        # image transforms (ImageNet compatible)
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_name: str) -> torch.Tensor:
        image_path = os.path.join(self.img_dir, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

    def _ingredients_to_multihot(self, ingredients: List[str]) -> torch.Tensor:
        vec = torch.zeros(self.num_ingredients, dtype=torch.float32)
        for ing in ingredients:
            if ing in self.ingredient_to_idx:
                vec[self.ingredient_to_idx[ing]] = 1.0
        return vec

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        image = self._load_image(sample["image"])
        ingredients_vec = self._ingredients_to_multihot(sample["ingredients"])

        return {
            "image": image,
            "ingredients": ingredients_vec,
        }
