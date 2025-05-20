from transformers import pipeline
import os
from PIL import Image

SCENIC_KEYWORDS = { … 同上 … }

class Tagger:
    def __init__(self):
        hf_token = os.environ.get("HFG_TOKEN")
        if not hf_token:
            raise ValueError("请在 Streamlit Secrets 配置 HFG_TOKEN")
        # 直接用 transformers pipeline
        self.pipe = pipeline(
            "image-classification",
            model="smilingwolf/wd-v1-4-vit-tagger-v2",
            token=hf_token,
            top_k=None,   # 先拿全量结果，再自己筛
        )

    def get_tags(self, img_path: str, top_k: int = 16, min_conf: float = 0.1):
        results = self.pipe(Image.open(img_path).convert("RGB"))
        scenic = [
            r["label"]
            for r in results
            if r["label"] in SCENIC_KEYWORDS and r["score"] >= min_conf
        ]
        return scenic[:top_k]
