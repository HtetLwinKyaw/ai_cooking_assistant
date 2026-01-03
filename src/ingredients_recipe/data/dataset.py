# src/ingredients_recipe/data/dataset.py
import os
import random
import json
from typing import List, Dict

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils.tokenizer import SimpleTokenizer

class RecipeDataset(Dataset):
    """
    Expects:
      data_dir/
        images/
        metadata.csv  (columns: image, ingredients, recipe)

    Builds:
      - ingredient2idx mapping (multi-label)
      - tokenizer for recipe text (word-level)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 224,
        max_recipe_len: int = 128,
        build_vocab: bool = False,
        min_ingredient_freq: int = 1,
        max_ingredients: int = None,
    ):
        """
        If build_vocab=True -> reads full metadata.csv and constructs vocabularies.
        For train/val/test splits you can call with the same data_dir and build_vocab=False
        after saving the mappings.
        """
        self.data_dir = data_dir
        self.split = split
        self.img_dir = os.path.join(data_dir, "images")
        self.meta_path = os.path.join(data_dir, "metadata.csv")
        self.img_size = img_size
        self.max_recipe_len = max_recipe_len

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip() if split == "train" else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # load metadata
        self.df = pd.read_csv(self.meta_path)
        # Optionally, you could filter splits here if metadata has split labels.
        # For simplicity the user can prepare separate metadata files per experiment.

        # build vocabularies
        if build_vocab:
            self._build_mappings(min_ingredient_freq, max_ingredients)
            self.tokenizer = SimpleTokenizer()
            self.tokenizer.build_vocab(self.df["recipe"].tolist())
            # save mappings
            torch.save({"ingredient2idx": self.ingredient2idx, "idx2ingredient": self.idx2ingredient}, os.path.join(data_dir, "ingredient_vocab.pt"))
            torch.save({"vocab": self.tokenizer.word2idx, "idx2word": self.tokenizer.idx2word}, os.path.join(data_dir, "recipe_vocab.pt"))
        else:
            # load mapping files
            vocab_path = os.path.join(data_dir, "recipe_vocab.pt")
            ing_path = os.path.join(data_dir, "ingredient_vocab.pt")
            ik = torch.load(ing_path)
            vk = torch.load(vocab_path)
            self.ingredient2idx = ik["ingredient2idx"]
            self.idx2ingredient = ik["idx2ingredient"]
            self.tokenizer = SimpleTokenizer()
            self.tokenizer.word2idx = vk["vocab"]
            self.tokenizer.idx2word = vk["idx2word"]
            self.tokenizer.vocab_size = len(self.tokenizer.word2idx)

        # prepare examples
        self.examples = []
        for _, row in self.df.iterrows():
            img = row["image"]
            ing = row["ingredients"]
            rc = row["recipe"]
            # ingredients -> list
            if isinstance(ing, float) and pd.isna(ing):
                ing_list = []
            else:
                ing_list = [x.strip().lower() for x in str(ing).split(",") if x.strip() != ""]
            self.examples.append({"image": img, "ingredients": ing_list, "recipe": str(rc)})

    def _build_mappings(self, min_freq=1, max_ingredients=None):
        # build ingredient vocabulary by frequency
        freq = {}
        for _, row in self.df.iterrows():
            s = row["ingredients"]
            if isinstance(s, float) and pd.isna(s):
                continue
            tokens = [x.strip().lower() for x in str(s).split(",") if x.strip() != ""]
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
        items = [k for k, v in freq.items() if v >= min_freq]
        items = sorted(items, key=lambda x: -freq[x])
        if max_ingredients:
            items = items[:max_ingredients]
        self.ingredient2idx = {w: i for i, w in enumerate(items)}
        self.idx2ingredient = {i: w for w, i in self.ingredient2idx.items()}

    def __len__(self):
        return len(self.examples)

    def _load_image(self, filename):
        path = os.path.join(self.img_dir, filename)
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def _ingredients_to_multihot(self, ing_list):
        vect = torch.zeros(len(self.ingredient2idx), dtype=torch.float32)
        for it in ing_list:
            if it in self.ingredient2idx:
                vect[self.ingredient2idx[it]] = 1.0
        return vect

    def __getitem__(self, idx):
        ex = self.examples[idx]
        image = self._load_image(ex["image"])
        ing_multi = self._ingredients_to_multihot(ex["ingredients"])
        # tokenize recipe -> ids (with bos/eos)
        token_ids = self.tokenizer.encode(ex["recipe"], max_length=self.max_recipe_len)
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        return {"image": image, "ingredients": ing_multi, "recipe_ids": token_ids}
