# src/ingredients_recipe/api/app.py

import io
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms

from ingredients_recipe.data.dataset import IngredientDataset
from ingredients_recipe.models.model import IngredientModel
from ingredients_recipe.recipes.generator import RecipeGenerator

app = FastAPI(title="AI Cooking Assistant API")

# -------------------------
# DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD DATASET (VOCAB ONLY)
# -------------------------
DATA_DIR = "data/raw"
dataset = IngredientDataset(data_dir=DATA_DIR)
ingredient_vocab = dataset.ingredient_vocab

# -------------------------
# LOAD INGREDIENT MODEL
# -------------------------
CHECKPOINT_PATH = "checkpoints/best.pt"

ingredient_model = IngredientModel(
    encoder_name="resnet50",
    num_ingredients=len(ingredient_vocab),
    pretrained_encoder=False,
    freeze_encoder=True,
).to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    ingredient_model.load_state_dict(checkpoint["model_state"])
else:
    ingredient_model.load_state_dict(checkpoint)

ingredient_model.eval()

# -------------------------
# LOAD RECIPE GENERATOR
# -------------------------
recipe_generator = RecipeGenerator()

# -------------------------
# IMAGE TRANSFORM
# -------------------------
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


def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor.to(device)


# -------------------------
# ROUTES
# -------------------------
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "device": str(device),
        "num_ingredients": len(ingredient_vocab),
    }


@app.post("/predict")
async def predict_ingredients_and_recipe(
    file: UploadFile = File(...),
    threshold: float = 0.5,
):
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)

    with torch.no_grad():
        logits = ingredient_model(image_tensor)
        probs = torch.sigmoid(logits).squeeze(0)

    predicted_ingredients = [
        ing
        for ing, p in zip(ingredient_vocab, probs)
        if p.item() >= threshold
    ]

    recipe = None
    if predicted_ingredients:
        recipe = recipe_generator.generate(predicted_ingredients)

    return {
        "ingredients": predicted_ingredients,
        "recipe": recipe,
        "threshold": threshold,
    }
