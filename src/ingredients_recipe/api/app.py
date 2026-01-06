# src/ingredients_recipe/api/app.py

import io
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms

from ingredients_recipe.data.dataset import IngredientDataset
from ingredients_recipe.models.model import IngredientModel
from ingredients_recipe.recipes.generator import RecipeGenerator
from ingredients_recipe.vision.dish_model import DishModel
from ingredients_recipe.postprocess.ingredient_rules import clean_ingredients

app = FastAPI(title="AI Cooking Assistant API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dish_model = DishModel(checkpoint_path="checkpoints/dish/best.pt")

DATA_DIR = "data/raw"
dataset = IngredientDataset(data_dir=DATA_DIR)
ingredient_vocab = dataset.ingredient_vocab

ING_CHECKPOINT = "checkpoints/best.pt"

ingredient_model = IngredientModel(
    encoder_name="resnet50",
    num_ingredients=len(ingredient_vocab),
    pretrained_encoder=False,
    freeze_encoder=True,
).to(device)

checkpoint = torch.load(ING_CHECKPOINT, map_location=device)
ingredient_model.load_state_dict(
    checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
)
ingredient_model.eval()

recipe_generator = RecipeGenerator()

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


def preprocess_image(image: Image.Image) -> torch.Tensor:
    return transform(image).unsqueeze(0).to(device)


@app.get("/")
def health_check():
    return {"status": "ok", "device": str(device)}


@app.post("/predict")
async def predict_ingredients_and_recipe(
    file: UploadFile = File(...),
    threshold: float = 0.5,
):
    threshold = max(0.0, min(1.0, threshold))

    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor = preprocess_image(pil_image)

    with torch.no_grad():
        logits = ingredient_model(image_tensor)
        probs = torch.sigmoid(logits).squeeze(0)

    predicted_ingredients = [
        ing for ing, p in zip(ingredient_vocab, probs)
        if p.item() >= threshold and ing != "sugar"
    ]

    dish_result = dish_model.predict(pil_image)

    dish = dish_result["top_prediction"]["dish"]
    dish_confidence = dish_result["top_prediction"]["confidence"]
    alternatives = dish_result["alternatives"]

    raw_ingredients = predicted_ingredients + [dish]

    final_ingredients = clean_ingredients(dish, raw_ingredients)
    final_ingredients = sorted(
        ing for ing in final_ingredients if ing != dish.lower()
    )

    recipe = recipe_generator.generate(final_ingredients)

    return {
        "dish": dish,
        "dish_confidence": dish_confidence,
        "dish_alternatives": alternatives,
        "ingredients": final_ingredients,
        "recipe": recipe,
        "threshold": threshold,
    }
