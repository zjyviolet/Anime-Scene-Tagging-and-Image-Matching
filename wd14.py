# wd14.py
from huggingface_hub import InferenceApi
from PIL import Image
import io

# 只保留你关心的风景词
SCENIC_KEYWORDS = {
    "sky","sea","forest","mountain","city","night","sunset",
    "trees","water","street","river","building","beach","clouds","sun","lake"
}

class Tagger:
    def __init__(self):
        # 指定模型 repo ID
        self.client = InferenceApi(
            repo_id="SmilingWolf/wd14-tagger",
            token=st.secrets["HFG_TOKEN"],
        )

    def get_tags(self, img_path, top_k=16, min_conf=0.1):
        # 读取成二进制
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        # 调用 HF API
        output = self.client(inputs=img_bytes)
        # output 是列表 e.g. [{"tag":"sky","score":0.98}, ...]
        scenic = [
            o["tag"] for o in output
            if o["tag"] in SCENIC_KEYWORDS and o["score"] >= min_conf
        ]
        return scenic[:top_k]
