# wd14.py

import os
from huggingface_hub import InferenceApi
from PIL import Image

# 只保留你关心的风景标签
SCENIC_KEYWORDS = {
    "sky","sea","forest","mountain","city","night","sunset",
    "trees","water","street","river","building",
    "beach","clouds","sun","lake"
}

class Tagger:
    def __init__(self):
        """
        从环境变量 HFG_TOKEN 读取 HF API Token。
        在 Streamlit Cloud 上请在 'Secrets' 中添加：
        HFG_TOKEN = "hf_xxx你的tokenxxx"
        """
        hf_token = os.environ.get("HFG_TOKEN", None)
        if not hf_token:
            raise ValueError("环境变量 HFG_TOKEN 未设置，请在 Streamlit Secrets 中添加。")
        # 指定 repo_id 和 token，InferenceApi 会把推理请求发到 HF 服务器
        self.client = InferenceApi(
            repo_id="SmilingWolf/wd14-tagger",
            token=hf_token,
        )

    def get_tags(self, img_path: str, top_k: int = 16, min_conf: float = 0.1) -> list[str]:
        """
        调用 Hugging Face Inference API 提取风景标签。
        """
        # 读取图片为 bytes
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        # API 返回类似 [{"tag":"sky","score":0.98}, ...]
        output = self.client(inputs=img_bytes)

        # 过滤并保留你要的风景标签
        scenic = [
            o["tag"]
            for o in output
            if o["tag"] in SCENIC_KEYWORDS and o["score"] >= min_conf
        ]
        return scenic[:top_k]
