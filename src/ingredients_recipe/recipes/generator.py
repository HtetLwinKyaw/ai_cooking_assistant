# src/ingredients_recipe/recipes/generator.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class RecipeGenerator:
    """
    Generates cooking recipes from ingredient lists using a pretrained FLAN-T5 model.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_length: int = 256,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        self.max_length = max_length
        self.model.eval()

    def generate(self, ingredients: list[str]) -> str:
        ingredient_text = ", ".join(sorted(set(ingredients)))

        prompt = (
            "Create a complete cooking recipe using the following ingredients.\n\n"
            f"Ingredients: {ingredient_text}\n\n"
            "Write numbered cooking steps."
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                min_length=80,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                num_beams=4,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

        recipe = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        # ---- SAFETY: remove prompt echo if it happens ----
        if "Ingredients:" in recipe:
            recipe = recipe.split("Ingredients:")[-1].strip()

        return recipe
