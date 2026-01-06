# src/ingredients_recipe/recipes/generator.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class RecipeGenerator:
    """
    Generates robust cooking recipes from ingredient lists
    using a large instruction-tuned model.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        max_length: int = 384,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        self.max_length = max_length
        self.model.eval()

    def generate(self, ingredients: list[str]) -> dict:
        ingredient_text = ", ".join(sorted(set(ingredients)))

        prompt = (
            "You are a professional chef.\n\n"
            "Write a complete, realistic cooking recipe using ONLY the ingredients below.\n\n"
            f"Ingredients: {ingredient_text}\n\n"
            "Write the recipe using realistic cooking methods (e.g., boiling, sautéing, simmering) appropriate for the dish."
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
                min_length=150,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                num_beams=4,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

        text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        # ---- Clean obvious junk tokens ----
        text = text.replace("<", "").replace(">", "").strip()

        # ---- Extract steps if possible ----
        steps = []
        for line in text.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                steps.append(line.lstrip("0123456789. ").strip())

        if not steps:
            steps = [
            s.strip()
            for s in text.split(". ")
            if len(s.strip()) > 20
    ]

        return {
            "recipe_text": text,
            "steps": steps,
        }
