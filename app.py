# app.py
import os
import csv
import random
import streamlit as st
from PIL import Image
from clip_tagger import CLIPTagger

# Page configuration
st.set_page_config(page_title="Anime → Scenic Tagging & Matching", layout="centered")
st.title("Anime Scene Tagging and Image Matching")

# Cache a single CLIPTagger instance
@st.cache_resource(show_spinner=False)
def get_tagger():
    return CLIPTagger(device="cpu")

tagger = get_tagger()

# Cache loading of CSV index
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

# Adjustable parameters
min_conf = st.slider(
    label="Minimum similarity threshold",
    min_value=0.0, max_value=1.0, value=0.25, step=0.05
)
top_k = st.number_input(
    label="Maximum number of tags",
    min_value=1, max_value=16, value=5, step=1
)

# Upload an anime image
uploaded_file = st.file_uploader("Upload an anime image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded anime image", use_container_width=True)

    # Save to disk for tagging
    temp_path = "temp.png"
    image.save(temp_path)

    # Extract tags
    with st.spinner("Extracting tags..."):
        tags = tagger.get_tags(temp_path, top_k=top_k, min_conf=min_conf)

    if tags:
        st.success("Detected scenic tags:")
        st.write(", ".join(tags))
    else:
        st.warning("No scenic tags detected. Try lowering the threshold or using a different image.")
        tags = []

    # Find matches in the index
    matched_images = [
        name for name, img_tags in index.items()
        if any(tag in img_tags for tag in tags)
    ]

    st.markdown("---")
    if matched_images:
        # 1. Show match count
        st.write(f"Found **{len(matched_images)}** matching images in the database.")

        # 2. Pick a random match
        if "selected_image" not in st.session_state:
            st.session_state.selected_image = random.choice(matched_images)

        # 3. Reset button for a new random match
        if st.button("Reset Random Match"):
            st.session_state.selected_image = random.choice(matched_images)

        selected = st.session_state.selected_image
        img_path = os.path.join("data/images", selected)

        # Display the matched image
        try:
            matched_img = Image.open(img_path).convert("RGB")
            st.image(matched_img, caption=selected, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to open image '{selected}': {e}")

        # 4. Show all tags that matched
        common_tags = [t for t in tags if t in index[selected]]
        if common_tags:
            st.write(f"Matched on tag(s): **{', '.join(common_tags)}**")
    else:
        st.info("No matching images found. Please check your CSV or try different tags.")
