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
        """
        ingredients: list of ingredient strings
        returns: generated recipe text
        """

        ingredient_text = ", ".join(ingredients)

        prompt = (
            "Write a clear, step-by-step cooking recipe using the following ingredients:\n"
            f"{ingredient_text}\n\n"
            "Include preparation steps and cooking instructions."
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
                num_beams=4,
                temperature=0.9,
            )

        recipe = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return recipe
