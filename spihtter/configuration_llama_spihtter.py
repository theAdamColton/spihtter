from transformers import LlamaConfig


class LlamaSpihtterConfig(LlamaConfig):
    def __init__(
        self,
        max_height: int = 128,
        max_width: int = 128,
        image_channels: int = 3,
        **kwargs,
    ):
        self.max_height = max_height
        self.max_width = max_width
        self.image_channels = image_channels
        super().__init__(**kwargs)
