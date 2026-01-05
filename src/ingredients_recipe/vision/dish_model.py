# src/ingredients_recipe/vision/dish_model.py

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from ingredients_recipe.vision.dish_labels import load_dish_labels


class DishModel:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.class_to_idx = checkpoint["class_to_idx"]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # ✅ SINGLE SOURCE OF TRUTH FOR DATA
        category_path = Path("data/uecfood_category.txt").resolve()

        if not category_path.exists():
            raise FileNotFoundError(
                f"UECFood category file not found at {category_path}"
            )

        self.dish_labels = load_dish_labels(str(category_path))

        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(
            self.model.fc.in_features, len(self.class_to_idx)
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def predict(self, image: Image.Image) -> str:
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        pred_idx = logits.argmax(dim=1).item()
        class_id = self.idx_to_class[pred_idx]
        return self.dish_labels.get(class_id, class_id)
