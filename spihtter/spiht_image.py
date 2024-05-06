import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

import spiht
from spiht import EncodingResult

from .spiht_configuration import SpihtConfiguration
from .utils import bits_to_bytes, bytes_to_bits, imload, imshow


@dataclass
class SpihtImage:
    height: int
    width: int
    max_n: int
    spiht_configuration: SpihtConfiguration
    _encoded_bytes: Optional[bytes]
    _encoded_bits: Optional[np.ndarray] = None

    @property
    def level(self):
        return self.spiht_configuration.get_level(self.height, self.width)

    @property
    def shape(self):
        return self.spiht_configuration.image_channels, self.height, self.width

    @property
    def encoded_bytes(self):
        if self._encoded_bytes is None:
            assert self._encoded_bits is not None
            bits = np.array(self._encoded_bits, dtype=np.uint8)
            self._encoded_bytes = bits_to_bytes(bits)
        return self._encoded_bytes

    @property
    def encoded_bits(self):
        if self._encoded_bits is None:
            if self._encoded_bytes is not None:
                self._encoded_bits = bytes_to_bits(self._encoded_bytes)
            else:
                return np.array([])
        return self._encoded_bits

    def decode(self, return_metadata=False):
        encoding_result = self._to_encoding_result()
        return spiht.decode_image(
            encoding_result,
            spiht_settings=self.spiht_configuration.spiht_settings,
            return_metadata=return_metadata,
        )

    def show(self):
        im = self.decode()
        imshow(im)

    @staticmethod
    def from_encoding_result(
        encoding_result: EncodingResult, spiht_configuration: SpihtConfiguration
    ):
        h = encoding_result.h
        w = encoding_result.w
        assert spiht_configuration.get_level(h, w) == encoding_result.level
        assert spiht_configuration.image_channels == encoding_result.c
        return SpihtImage(
            height=h,
            width=w,
            max_n=encoding_result.max_n,
            spiht_configuration=spiht_configuration,
            _encoded_bytes=encoding_result.encoded_bytes,
        )

    def _to_encoding_result(self):
        encoding_result = EncodingResult(
            h=self.height,
            w=self.width,
            c=self.spiht_configuration.image_channels,
            max_n=self.max_n,
            level=self.spiht_configuration.get_level(self.height, self.width),
            encoded_bytes=self.encoded_bytes,
            _encoding_version=spiht.ENCODER_DECODER_VERSION,
        )
        return encoding_result

    def to_html_opening_tag(self):
        html_attrs = self.spiht_configuration.format_spiht_tag_attrs(
            dict(h=self.height, w=self.width, n=self.max_n)
        )
        return f"<spiht {html_attrs}>"

    @staticmethod
    def from_html_opening_tag_attrs(attrs, spiht_configuration: SpihtConfiguration):
        attrs = spiht_configuration.parse_spiht_tag_attrs(attrs)
        return SpihtImage(
            height=attrs["h"],
            width=attrs["w"],
            max_n=attrs["n"],
            spiht_configuration=spiht_configuration,
            _encoded_bits=None,
            _encoded_bytes=None,
        )

    def get_metadata(self):
        """
        Uses the fast decoder to get metadata ids
        """
        encoding_result = self._to_encoding_result()

        metadata = spiht.spiht_wrapper.decode_rec_array(
            encoding_result,
            self.spiht_configuration.spiht_settings,
            return_metadata=True,
        )["spiht_metadata"]
        metadata = torch.from_numpy(metadata).to(torch.long)
        return metadata

    def __len__(self):
        if self._encoded_bytes is not None:
            return len(self._encoded_bytes) * 8
        elif self._encoded_bits is not None:
            return len(self._encoded_bits)
        else:
            return 0

    @staticmethod
    def from_file(file, config: SpihtConfiguration, max_bits=25_000):
        image = imload(file)
        c, h, w = image.shape
        encoding_result = spiht.encode_image(
            image,
            config.spiht_settings,
            max_bits=max_bits,
            level=config.get_level(h, w),
        )
        return SpihtImage.from_encoding_result(encoding_result, config)
