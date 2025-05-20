# wd14.py
import os
from PIL import Image
from transformers import pipeline

# 只保留这些风景关键词
SCENIC_KEYWORDS = {
    "sky", "sea", "forest", "mountain", "city", "night", "sunset",
    "trees", "water", "night sky", "street", "river", "building",
    "beach", "clouds", "sun", "lake"
}

class Tagger:
    def __init__(self):
        # 从环境变量或 Streamlit Secrets 读取你的 HF Token
        hf_token = os.environ.get("HFG_TOKEN", "")
        if not hf_token:
            raise ValueError("请在 Streamlit Secrets 或环境变量中设置 HFG_TOKEN")
        # 用 transformers 的 pipeline，指定 image-classification
        self.pipe = pipeline(
            "image-classification",
            model="smilingwolf/wd-v1-4-vit-tagger-v2",
            token=hf_token,
            # 不要在这里设 top_k，拿全量结果再自己切
            top_k=None,
        )

    def get_tags(self, img_path: str, top_k: int = 16, min_conf: float = 0.1) -> list[str]:
        """
        Args:
          img_path: 图片文件路径
          top_k: 最多返回多少个标签
          min_conf: 置信度阈值
        Returns:
          List[str]：只包含 SCENIC_KEYWORDS 中的标签
        """
        # 直接传 PIL.Image 对象给 pipeline
        img = Image.open(img_path).convert("RGB")
        results = self.pipe(img)
        # 过滤出我们关心的风景词
        scenic = [
            r["label"]
            for r in results
            if r["label"] in SCENIC_KEYWORDS and r["score"] >= min_conf
        ]
        # 截断到 top_k
        return scenic[:top_k]
