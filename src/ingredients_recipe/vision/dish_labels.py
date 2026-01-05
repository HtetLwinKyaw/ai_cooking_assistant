# src/ingredients_recipe/vision/dish_labels.py

from pathlib import Path


def load_dish_labels(category_file: str) -> dict[str, str]:
    """
    Load UECFood category file.
    Returns: { "84": "spaghetti", ... }
    """
    mapping = {}

    with open(category_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue

            class_id, dish_name = parts
            mapping[class_id] = dish_name.replace("_", " ")

    return mapping
