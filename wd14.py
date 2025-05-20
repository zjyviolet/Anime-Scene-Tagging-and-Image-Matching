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
        # 指定 task="image-tagging"
        self.client = InferenceApi(
            repo_id="smilingwolf/wd-v1-4-vit-tagger-v2",
            token=hf_token,
            task="image-tagging",
        )

    def get_tags(self, img_path: str, top_k: int = 16, min_conf: float = 0.1):
        # 读取图片 bytes
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        # 调用推理
        output = self.client(inputs=img_bytes)
        # output 是 list of {"tag":..., "score":...}
        scenic = [
            o["tag"] for o in output
            if o["tag"] in SCENIC_KEYWORDS and o["score"] >= min_conf
        ]
        return scenic[:top_k]
