# wd14.py

from wdtagger import Tagger as WDTagger
from PIL import Image

class Tagger:
    def __init__(self, tags_csv: str = "my_selected_tags.csv"):
        """
        初始化 WD14 Tagger，只加载指定的标签子集以减少内存占用。
        
        :param tags_csv: 一行一个标签的 CSV 文件路径，只加载这里列出的标签。
        """
        # selected_tags 参数指定只读取这份 CSV 中的标签
        self.model = WDTagger(selected_tags=tags_csv)

    def get_tags(self, img_path: str, top_k: int = 20, min_conf: float = 0.1) -> list[str]:
        """
        从图片中提取标签。

        :param img_path: 图片文件的本地路径
        :param top_k: 最多返回多少个标签
        :param min_conf: 最低置信度阈值（映射到模型的 general_threshold）
        :return: 检测到的标签列表
        """
        # 载入并转换图片
        img = Image.open(img_path).convert("RGB")
        # 调用模型打标签，只会计算 tags_csv 中的那部分标签
        result = self.model.tag(img, general_threshold=min_conf)
        # result.general_tag_data 是一个有序字典 {标签: 置信度}
        tags = list(result.general_tag_data.keys())
        return tags[:top_k]
