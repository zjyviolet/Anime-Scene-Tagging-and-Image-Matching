# app.py
import os
import csv
import random
import streamlit as st
from PIL import Image
from wd14 import Tagger

# Page configuration
st.set_page_config(page_title="Anime → Scenic Tagging & Matching", layout="centered")

st.title("🎨 Anime Scene Tagging and Image Matching")

# Instantiate the tag extractor
tagger = Tagger()

# Load the image-tag index CSV once and cache it
@st.cache_data(show_spinner=False)
def load_index(csv_path="data/image-tag.csv"):
    """
    Load mapping: image_name -> [tag1, tag2, ...]
    CSV format: image_name,tags  (tags separated by semicolons)
    """
    index = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            name = row["image_name"]
            tags = [t.strip() for t in row["tags"].split(";") if t.strip()]
            index[name] = tags
    return index

index = load_index()

# Upload an anime image
uploaded_file = st.file_uploader("Upload an anime image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded anime image", use_container_width=True)

    # Extract scenic tags
    with st.spinner("Extracting scenic tags..."):
        image.save("temp.png")
        tags = tagger.get_tags("temp.png", top_k=20, min_conf=0.1)

    if tags:
        st.success("Detected scenic tags:")
        st.write(", ".join(tags))
    else:
        st.warning("No scenic tags detected. Try lowering the threshold or using a different image.")
        tags = []

    # Match extracted tags against the index
    matched_images = [
        name for name, img_tags in index.items()
        if any(tag in img_tags for tag in tags)
    ]

    st.markdown("---")
    if matched_images:
        # 1. Show count of matches
        st.write(f"Found **{len(matched_images)}** matching images in the database.")

        # 2 & 3. Pick one at random and allow regenerating
        if "selected_image" not in st.session_state:
            st.session_state.selected_image = random.choice(matched_images)

        if st.button("Reset a Random Match"):
            st.session_state.selected_image = random.choice(matched_images)

        selected = st.session_state.selected_image
        img_path = os.path.join("data/images", selected)

        # Display the selected image
        try:
            matched_img = Image.open(img_path).convert("RGB")
            st.image(matched_img, caption=selected, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to open image '{selected}': {e}")

        # 4. Show which tags matched (preserve order & show all)
    common_tags = [t for t in tags if t in index[selected]]
    if common_tags:
        st.write(f"Matched on tag(s): **{', '.join(common_tags)}**")

    else:
        st.info("⚠️ No matching images found. Please check your CSV or try different tags.")
