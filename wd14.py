# wd14.py
from wdtagger import Tagger as WDTagger
from PIL import Image

# 你关心的风景关键词集合
SCENIC_KEYWORDS = {
    "sky", "sea", "forest", "mountain", "city", "night", "sunset",
    "trees", "water", "street", "river", "building",
    "beach", "clouds", "sun", "lake"
}

class Tagger:
    def __init__(self, model_name: str = None):
        """
        初始化 WD14 Tagger。
        如果提供了 model_name，则会尝试加载指定模型，否则加载默认模型。
        """
        if model_name:
            # 一般 wdtagger 支持传入 model 参数来选择不同预训练模型
            self.model = WDTagger(model=model_name)
        else:
            self.model = WDTagger()

    def get_tags(
        self,
        img_path: str,
        top_k: int = 16,
        min_conf: float = 0.1
    ) -> list[str]:
        """
        从本地图片中提取风景标签。

        Args:
            img_path: 图片文件路径
            top_k: 最多返回几个风景标签
            min_conf: 置信度阈值，只有置信度 >= min_conf 的标签才会被考虑

        Returns:
            List[str]: 筛选后的风景关键词列表，长度最多为 top_k
        """
        # 打开并转成 RGB
        img = Image.open(img_path).convert("RGB")

        # 只要一般标签，字符标签阈值设为 1.0 跳过所有人物类标签
        result = self.model.tag(
            img,
            general_threshold=min_conf,
            character_threshold=1.0
        )

        # result.general_tag_data: OrderedDict{标签: 置信度}
        # 筛选出风景相关关键词
        scenic_tags = [
            tag for tag in result.general_tag_data.keys()
            if tag in SCENIC_KEYWORDS
        ]

        # 返回最多 top_k 个
        return scenic_tags[:top_k]
