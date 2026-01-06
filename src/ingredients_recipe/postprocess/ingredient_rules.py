# src/ingredients_recipe/postprocess/ingredient_rules.py

from typing import List


# Ingredients that almost never belong to certain dishes
DISALLOWED_BY_DISH = {
    "spaghetti": {"flour", "egg"},
    "spaghetti meat sauce": {"flour", "egg"},
    "pizza": {"egg"},
    "fried rice": {"flour"},
}


# Ingredients that should almost always be present
REQUIRED_BY_DISH = {
    "spaghetti": {"spaghetti"},
    "spaghetti meat sauce": {"spaghetti", "meat sauce"},
    "pizza": {"pizza dough"},
}


def clean_ingredients(
    dish: str,
    ingredients: List[str],
) -> List[str]:
    dish = dish.lower()

    ingredients = {i.lower() for i in ingredients}

    # Remove disallowed ingredients
    for key, banned in DISALLOWED_BY_DISH.items():
        if key in dish:
            ingredients -= banned

    # Add required ingredients
    for key, required in REQUIRED_BY_DISH.items():
        if key in dish:
            ingredients |= required

    return sorted(ingredients)
