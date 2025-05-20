from huggingface_hub import InferenceApi
import os
from PIL import Image

SCENIC_KEYWORDS = {
    "sky","sea","forest","mountain","city","night","sunset",
    "trees","water","night sky","street","river","building",
    "beach","clouds","sun","lake"
}

class Tagger:
    def __init__(self):
        hf_token = os.environ.get("HFG_TOKEN")
        if not hf_token:
            raise ValueError("请在 Streamlit Secrets 里配置 HFG_TOKEN")
        # 这里把 task 改成 image-classification
        self.client = InferenceApi(
            repo_id="smilingwolf/wd-v1-4-vit-tagger-v2",
            token=hf_token,
            task="image-classification",
        )

    def get_tags(self, img_path: str, top_k: int = 16, min_conf: float = 0.1):
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        # 结果是 [{"label":"sky","score":0.9}, ...]
        output = self.client(img_bytes)
        scenic = [
            o["label"]
            for o in output
            if o["label"] in SCENIC_KEYWORDS and o["score"] >= min_conf
        ]
        return scenic[:top_k]
