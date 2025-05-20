# wd14.py
import os
from huggingface_hub import InferenceApi
from PIL import Image

SCENIC_KEYWORDS = {
    "sky","sea","forest","mountain","city","night","sunset",
    "trees","water","night sky","street","river","building",
    "beach","clouds","sun","lake"
}

class Tagger:
    def __init__(self):
        hf_token = os.environ.get("HFG_TOKEN") or ""
        if not hf_token:
            raise ValueError("请在 Streamlit Secrets 或环境变量中设置 HFG_TOKEN")
        self.client = InferenceApi(
            repo_id="smilingwolf/wd-v1-4-vit-tagger-v2",
            token=hf_token,
            task="image-classification",
        )

    def get_tags(self, img_path: str, top_k: int = 16, min_conf: float = 0.1):
        # 以二进制模式打开文件，让 requests 做 multipart/form-data 上传
        with open(img_path, "rb") as f:
            output = self.client(files={"file": f})
        # output: [{"label":"sky","score":0.92}, ...]
        scenic = [
            item["label"]
            for item in output
            if item["label"] in SCENIC_KEYWORDS and item["score"] >= min_conf
        ]
        return scenic[:top_k]
