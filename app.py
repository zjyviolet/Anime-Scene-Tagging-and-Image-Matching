# app.py
import os
import csv
import random
import streamlit as st
from PIL import Image
from clip_tagger import CLIPTagger

# 页面配置
st.set_page_config(page_title="Anime → Scenic Tagging & Matching", layout="centered")
st.title("🎨 Anime Scene Tagging and Image Matching")

# 缓存一次 CLIPTagger 实例
@st.cache_resource(show_spinner=False)
def get_tagger():
    return CLIPTagger(device="cpu")

tagger = get_tagger()

# 缓存加载 CSV 索引
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

# 参数：阈值和最多标签数
min_conf = st.slider("最低相似度阈值", 0.0, 1.0, 0.25, 0.05)
top_k = st.number_input("最多标签数", min_value=1, max_value=16, value=5, step=1)

# 上传动漫图
uploaded_file = st.file_uploader("Upload an anime image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    # 显示上传的图
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded anime image", use_container_width=True)

    # 保存到本地临时文件，再做预测
    temp_path = "temp.png"
    image.save(temp_path)

    with st.spinner("Extracting tags..."):
        tags = tagger.get_tags(temp_path, top_k=top_k, min_conf=min_conf)

    if tags:
        st.success("Detected scenic tags:")
        st.write(", ".join(tags))
    else:
        st.warning("No scenic tags detected. Try lowering the threshold or using a different image.")
        tags = []

    # 在索引里匹配
    matched_images = [
        name for name, img_tags in index.items()
        if any(tag in img_tags for tag in tags)
    ]

    st.markdown("---")
    if matched_images:
        # 1. 匹配数量
        st.write(f"Found **{len(matched_images)}** matching images in the database.")

        # 2. 随机挑一张
        if "selected_image" not in st.session_state:
            st.session_state.selected_image = random.choice(matched_images)

        # 3. 重置按钮
        if st.button("Reset a Random Match"):
            st.session_state.selected_image = random.choice(matched_images)

        selected = st.session_state.selected_image
        img_path = os.path.join("data/images", selected)

        # 显示选中的真实图
        try:
            matched_img = Image.open(img_path).convert("RGB")
            st.image(matched_img, caption=selected, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to open image '{selected}': {e}")

        # 4. 显示所有匹配的标签
        common_tags = [t for t in tags if t in index[selected]]
        if common_tags:
            st.write(f"Matched on tag(s): **{', '.join(common_tags)}**")
    else:
        st.info("⚠️ No matching images found. Please check your CSV or try different tags.")
