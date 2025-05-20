# wd14.py
from wdtagger import Tagger as WDTagger
from PIL import Image

# Set of scenic keywords to filter the model's output
SCENIC_KEYWORDS = {
    "sky", "sea", "forest", "mountain", "city", "night", "sunset",
    "trees", "water", "night sky", "street", "river", "building",
    "beach", "clouds", "sun", "lake"
}

class Tagger:
    def __init__(self):
        """Load the WD14 Tagger model (from local cache if offline)."""
        self.model = WDTagger()

    def get_tags(self, img_path: str, top_k: int = 20, min_conf: float = 0.1) -> list[str]:
        """
        Extract scenic tags from an image.

        Args:
            img_path: Path to the image file.
            top_k: Maximum number of tags to return.
            min_conf: Minimum confidence threshold for filtering tags.

        Returns:
            A list of scenic keywords detected in the image.
        """
        img = Image.open(img_path).convert("RGB")
        # Pass the confidence threshold to the model's general_threshold
        result = self.model.tag(img, general_threshold=min_conf)
        # result.general_tag_data is an OrderedDict[tag -> confidence], sorted by confidence
        filtered = [
            tag
            for tag in result.general_tag_data.keys()
            if tag in SCENIC_KEYWORDS
        ]
        return filtered[:top_k]
