# clip_tagger.py
import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# 我们关心的 16 个风景关键词
SCENIC_KEYWORDS = [
    "sky", "sea", "forest", "mountain", "city", "night", "sunset",
    "trees", "water", "street", "river", "building", "beach",
    "clouds", "sun", "lake"
]

class CLIPTagger:
    def __init__(self, device: str = "cpu"):
        """
        加载 CLIP 模型与 processor，只运行一次。
        """
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # 文本输入一次性预处理：16 个关键词
        # 会得到 shape (16, dim_text)
        text_inputs = self.processor(text=SCENIC_KEYWORDS, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**{
                k: v.to(self.device) for k, v in text_inputs.items()
            })
        # 归一化
        self.text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

    def get_tags(self, img_path: str, top_k: int = 5, min_conf: float = 0.2) -> list[str]:
        """
        对一张图算和每个关键词的相似度，返回符合阈值的 top_k 标签。
        Args:
          img_path: 图片路径
          top_k: 最多返回几个关键词
          min_conf: 相似度阈值（cosine similarity 大于等于此值才算匹配）
        """
        # 读取并预处理图片
        image = Image.open(img_path).convert("RGB")
        img_inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            img_emb = self.model.get_image_features(**{
                k: v.to(self.device) for k, v in img_inputs.items()
            })
        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)  # 归一化，shape (1, dim)

        # 计算相似度：(1, dim) @ (16, dim).T -> (1,16)
        sims = (img_emb @ self.text_embeddings.T).squeeze(0)  # (16,)

        # 转为 CPU 上的 numpy，筛选大于阈值的
        sims = sims.cpu()
        # 排序，取 top_k
        topk_vals, topk_idx = torch.topk(sims, k=top_k)
        tags = []
        for val, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
            if val >= min_conf:
                tags.append(SCENIC_KEYWORDS[idx])
        return tags
