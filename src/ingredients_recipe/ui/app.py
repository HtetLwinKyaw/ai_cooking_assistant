# src/ingredients_recipe/ui/app.py

import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="AI Cooking Assistant",
    page_icon="🍳",
    layout="centered",
)

st.title("🍳 AI Cooking Assistant")
st.write("Upload a food image to get ingredients and a cooking recipe.")

threshold = st.slider(
    "Ingredient confidence threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.05,
)

uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Recipe"):
        with st.spinner("Analyzing image and generating recipe..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }

            params = {"threshold": threshold}

            try:
                response = requests.post(
                    API_URL,
                    files=files,
                    params=params,
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()

                ingredients = result.get("ingredients", [])
                recipe = result.get("recipe")

                if ingredients:
                    st.subheader("🧂 Predicted Ingredients")
                    for ing in ingredients:
                        st.markdown(f"- **{ing}**")
                else:
                    st.warning("No ingredients detected. Try lowering the threshold.")

                if recipe:
                    st.subheader("📖 Generated Recipe")
                    st.write(recipe)
                else:
                    st.warning("Recipe could not be generated.")

            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
