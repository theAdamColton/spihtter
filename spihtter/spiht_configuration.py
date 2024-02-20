"""
Base class BaseSpihtConfiguration defines some parameters and functions
that are used to encode/decode images to bits


All configuration classes follow a naming pattern ending with the suffix 'SpihtConfiguration'
"""
import math
from typing import List, Optional, Tuple
from dataclasses import dataclass
import sys

from spiht import SpihtSettings

from spihtter.utils import bits_to_int, int_to_bits


def get_configuration(name:str, **kwargs):
    curr_module =  sys.modules[__name__]
    names = [s for s in dir(curr_module) if s.endswith("SpihtConfiguration")]
    if name in names:
        config_class = getattr(curr_module, name)
        return config_class(**kwargs)
    raise ValueError(f"config name {name} doesn't exist! Available config names are {names}")


@dataclass
class BaseSpihtConfiguration:
    """
    This class contains the configuration used by the spiht algorithm

    The spiht algorithm doesn't require a set-in-stone number of channels, or
    number of levels. This class stores the settings required to specify and
    calculate these parameters.

    The default settings are designed to work well for natural, RGB colored
    images.

    The height and width can also be expressed as hardcoded numbers
    """

    wavelet: str = "bior2.2"
    quantization_scale: float = 150.0
    color_model: Optional[str] = "Jzazbz"
    mode: str = "symmetric"
    per_channel_quant_scales: Optional[Tuple[float, float, float]] = (8, 1, 1)
    image_channels: int = 3
    height: Optional[int] = None
    width: Optional[int] = None

    def spiht_tag_attr_keys(self):
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
        return {
            k: self.parse_spiht_attr_number(attrs[k])
            for k in self.spiht_tag_attr_keys()
        }

    def parse_spiht_attr_number(self, x: str):
        """
        This determines how number attributes in the spiht html tag are parsed
        """
        return int(x)

    def format_spiht_attr_number(self, x: int):
        """
        This determines how number attributes in the spiht html tag are formatted
        """
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
        return math.floor(min(math.log2(h / 4), math.log2(w / 4)))

    def __post_init__(self):
        self.spiht_settings = SpihtSettings(
            wavelet=self.wavelet,
            quantization_scale=self.quantization_scale,
            mode=self.mode,
            color_model=self.color_model,
            per_channel_quant_scales=self.per_channel_quant_scales,
        )


@dataclass
class MnistSpihtConfiguration(BaseSpihtConfiguration):
    """
    Mnist images are 28 by 28 and have only one channel.

    You can save a lot of bits and improve reconstruction quality by using
    special settings.
    """

    wavelet: str = "bior1.1"
    quantization_scale: float = 50.0
    color_model: Optional[str] = None
    per_channel_quant_scales: Optional[Tuple[float, float, float]] = None
    image_channels: int = 1

    def spiht_tag_attr_keys(self):
        return ["n"]

    def parse_spiht_tag_attrs(self, attrs: dict):
        attrs = super().parse_spiht_tag_attrs(attrs)
        attrs["h"] = 28
        attrs["w"] = 28
        return attrs

    def get_level(self, h, w):
        return 3

    def parse_spiht_attr_number(self, x: str):
        """
        This determines how number attributes in the spiht html tag are parsed
        """
        return bits_to_int(x)

    def format_spiht_attr_number(self, x: int):
        """
        This determines how number attributes in the spiht html tag are formatted
        """
        return int_to_bits(x)


@dataclass
class CifarSpihtConfiguration(BaseSpihtConfiguration):
    quantization_scale: float = 10.0
    wavelet: str = "bior4.4"
    color_model: Optional[str] = "IPT"
    mode: str = "periodization"

    def get_level(self, h: int, w: int):
        return 4

    def spiht_tag_attr_keys(self):
        return ["n"]

    def parse_spiht_tag_attrs(self, attrs: dict):
        attrs = super().parse_spiht_tag_attrs(attrs)
        attrs["h"] = 32
        attrs["w"] = 32
        return attrs

    def parse_spiht_attr_number(self, x: str):
        """
        This determines how number attributes in the spiht html tag are parsed
        """
        return bits_to_int(x)

    def format_spiht_attr_number(self, x: int):
        """
        This determines how number attributes in the spiht html tag are formatted
        """
        return int_to_bits(x)


@dataclass
class TinyImagenetSpihtConfiguration(BaseSpihtConfiguration):
    quantization_scale: float = 20.0
    wavelet: str = "bior4.4"
    color_model: Optional[str] = "IPT"
    mode: str = "periodization"

    def get_level(self, h: int, w: int):
        return 5

    def spiht_tag_attr_keys(self):
        return ["n"]

    def parse_spiht_tag_attrs(self, attrs: dict):
        attrs = super().parse_spiht_tag_attrs(attrs)
        attrs["h"] = 64
        attrs["w"] = 64
        return attrs

    def parse_spiht_attr_number(self, x: str):
        """
        This determines how number attributes in the spiht html tag are parsed
        """
        return bits_to_int(x)

    def format_spiht_attr_number(self, x: int):
        """
        This determines how number attributes in the spiht html tag are formatted
        """
        return int_to_bits(x)


@dataclass
class SDVaeSpihtConfiguration(BaseSpihtConfiguration):
    quantization_scale: float = 50.
    wavelet: str = "db2"
    per_channel_quant_scales: Optional[Tuple[float, float, float]] = None
    image_channels: int = 4
    color_model: Optional[str] = None
