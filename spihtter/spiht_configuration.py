"""
Base class BaseSpihtConfiguration defines some parameters and functions
that are used to encode/decode images to bits


All configuration classes follow a naming pattern ending with the suffix 'SpihtConfiguration'
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass

from spiht import SpihtSettings

from spihtter.utils import bits_to_int, int_to_bits


@dataclass
class SpihtConfiguration:
    """
    This class contains the configuration used by the spiht algorithm
     and the configuration used to format the spiht parameters as html attrs

    The spiht algorithm doesn't require a set-in-stone number of channels, or
    number of levels. This class stores the settings required to specify and
    calculate these parameters.

    The default settings are designed to work well for natural, RGB colored
    images.

    The height and width can also be expressed as hardcoded numbers using `height_width_fixed`

    The level can be specified as a fixed value using `level_fixed`
    """

    wavelet: str = "bior2.2"
    quantization_scale: float = 150.0
    color_model: Optional[str] = "Jzazbz"
    mode: str = "symmetric"
    level_fixed: Optional[int] = None
    format_attrs_as_bits: bool = False
    height_width_fixed: Optional[Tuple[int, int]] = None
    per_channel_quant_scales: Optional[Tuple[float, float, float]] = (8, 1, 1)
    image_channels: int = 3
    off_bit_token: str = "\x00"
    on_bit_token: str = "\x01"

    def spiht_tag_attr_keys(self):
        """
        returns a list of the attribute keys that must appear in the spiht html
        """
        if self.height_width_fixed:
            return ["n"]

        return ["h", "w", "n"]

    def format_spiht_tag_attrs(self, attrs: dict):
        """
        returns a string which is how the attrs are encoded as html attributes
        """
        strings = [
            f"{k}={self.format_spiht_attr_number(attrs[k])}"
            for k in self.spiht_tag_attr_keys()
        ]
        return " ".join(strings)

    def parse_spiht_tag_attrs(self, attrs: dict):
        """
        parses integers from the attrs dict
        """
        d = {
            k: self.parse_spiht_attr_number(attrs[k])
            for k in self.spiht_tag_attr_keys()
        }

        if self.height_width_fixed:
            d["h"], d["w"] = self.height_width_fixed

        return d

    def parse_spiht_attr_number(self, x: str):
        """
        This determines how number attributes in the spiht html tag are parsed
        """
        if self.format_attrs_as_bits:
            return bits_to_int(x)
        return int(x)

    def format_spiht_attr_number(self, x: int):
        """
        This determines how number attributes in the spiht html tag are formatted
        """
        if self.format_attrs_as_bits:
            return int_to_bits(x)
        return str(x)

    def get_level(self, h: int, w: int):
        """
        h: Image height
        w: Image width

        by default, the level is calculated so that the top level coeffs will
        have a height and width of 4. (Assuming no padding is added)

        It usually helps reconstruction quality to have a smaller top level.
        It also theoretically removes some of the spacial bias that can harm
        auto-regressive models.
        """
        if self.level_fixed is not None:
            return self.level_fixed

        return math.floor(min(math.log2(h / 4), math.log2(w / 4)))

    def __post_init__(self):
        self.spiht_settings = SpihtSettings(
            wavelet=self.wavelet,
            quantization_scale=self.quantization_scale,
            mode=self.mode,
            color_model=self.color_model,
            per_channel_quant_scales=self.per_channel_quant_scales,
        )
