# src/ingredients_recipe/vision/dish_classifier.py

import torch
import open_clip
from PIL import Image


class DishClassifier:
    """
    Zero-shot dish classifier using CLIP.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model = self.model.to(self.device).eval()

        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        # Common food dishes (expandable)
        self.dish_labels = [
            "spaghetti",
            "pasta",
            "pizza",
            "burger",
            "fried rice",
            "noodles",
            "salad",
            "sandwich",
            "soup",
            "steak",
            "curry",
            "sushi",
        ]

        self.text_tokens = self.tokenizer(
            [f"a photo of {dish}" for dish in self.dish_labels]
        ).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> str:
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        image_features = self.model.encode_image(image_tensor)
        text_features = self.model.encode_text(self.text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).softmax(dim=-1)
        best_idx = similarity.argmax(dim=-1).item()

        return self.dish_labels[best_idx]
