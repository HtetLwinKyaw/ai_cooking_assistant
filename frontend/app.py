import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://localhost:8000/predict"

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Cooking Assistant",
    page_icon="🍝",
    layout="centered",
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🍽️ AI Cooking Assistant")
st.markdown(
    "Upload a food image and get the **dish name**, **ingredients**, and a **generated recipe**."
)

st.divider()

# -------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"],
)

threshold = st.slider(
    "Ingredient confidence threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.05,
)

# -------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Food"):
        with st.spinner("Analyzing image..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }

            response = requests.post(
                API_URL,
                files=files,
                params={"threshold": threshold},
                timeout=120,
            )

        if response.status_code != 200:
            st.error("❌ Failed to get response from API")
        else:
            data = response.json()

            # -------------------------------------------------
            # DISH RESULT
            # -------------------------------------------------
            st.subheader("🍲 Predicted Dish")
            st.markdown(
                f"**{data['dish']}**  \n"
                f"Confidence: `{data['dish_confidence']:.2f}`"
            )

            if data.get("dish_alternatives"):
                with st.expander("Other possible dishes"):
                    for alt in data["dish_alternatives"]:
                        st.markdown(
                            f"- {alt['dish']} ({alt['confidence']:.2f})"
                        )

            # -------------------------------------------------
            # INGREDIENTS
            # -------------------------------------------------
            st.subheader("🧂 Ingredients")
            st.write(", ".join(data["ingredients"]))

            # -------------------------------------------------
            # RECIPE
            # -------------------------------------------------
            st.subheader("📖 Recipe")

            recipe = data["recipe"]

            if isinstance(recipe, dict):
                st.markdown(recipe.get("recipe_text", ""))

                steps = recipe.get("steps", [])
                if steps:
                    st.markdown("**Steps:**")
                    for i, step in enumerate(steps, 1):
                        st.markdown(f"{i}. {step}")
            else:
                st.markdown(recipe)

            st.success("✅ Done!")
