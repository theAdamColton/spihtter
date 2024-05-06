import random
from dataclasses import dataclass
from typing import Optional
import numpy as np
from torchvision import transforms
import webdataset as wds
import spiht

from spihtter.process_inputs import SpihtInputProcessor
from spihtter.spiht_configuration import SpihtConfiguration
from spihtter.spiht_image import SpihtImage
from spihtter.utils import pad_truncate_to


class _SpihtImagePreprocessor:
    """
    expects rows with pixel_values
    pixel_values are h,w,c uint8 np ndarrays
    """

    def __init__(
        self,
        spiht_configuration: SpihtConfiguration,
        max_seq_len: Optional[int],
        bpp: Optional[float],
    ):
        self.spiht_configuration = spiht_configuration
        self.max_seq_len = max_seq_len
        self.bpp = bpp
        assert max_seq_len or bpp

    def __call__(self, row: dict):
        conf = self.spiht_configuration
        max_seq_len = self.max_seq_len

        pixel_values = row.pop("pixel_values")

        if pixel_values.ndim == 3:
            pixel_values = np.moveaxis(pixel_values, -1, 0)
        else:
            pixel_values = pixel_values[None, ...]

        assert pixel_values.dtype == np.uint8

        pixel_values = pixel_values / 255

        c, h, w = pixel_values.shape

        assert (
            c == self.spiht_configuration.image_channels
        ), f"{c} != {self.spiht_configuration.image_channels}"

        max_bits = self.max_seq_len
        if max_bits is None:
            max_bits = int(h * w * self.bpp)
        encoding_result = spiht.encode_image(
            pixel_values,
            spiht_settings=conf.spiht_settings,
            level=conf.get_level(h, w),
            max_bits=max_seq_len,
        )

        d = encoding_result.to_dict()
        d.update(row)
        return d


class _SpihtHtmlFormatter:
    def __init__(self, input_processor: SpihtInputProcessor, max_seq_len: int):
        self.input_processor = input_processor
        self.max_seq_len = max_seq_len

    def __call__(self, row: dict):
        conf = self.input_processor.spiht_configuration
        input_processor = self.input_processor

        if "encoding_result" in row:
            encoding_result = row["encoding_result"]
            assert isinstance(encoding_result, spiht.EncodingResult)
        else:
            encoding_result = spiht.EncodingResult.from_dict(row)

        spiht_image = SpihtImage.from_encoding_result(encoding_result, conf)

        label = row.pop("label")
        text = f"{label}"
        input_ids, metadata_ids = input_processor.process_normalized_images_texts(
            images=[None, spiht_image], texts=[text, None]
        )

        input_ids = pad_truncate_to(
            input_ids, self.max_seq_len, input_processor.tokenizer.pad_token_id
        )
        spiht_metadata = pad_truncate_to(metadata_ids, self.max_seq_len, 0)

        return dict(
            input_ids=input_ids,
            spiht_metadata_ids=spiht_metadata,
        )


@dataclass
class DatasetArgs:
    dataset_type: str = "wds-image"  # or 'wds-preprocessed'
    dataset: str = ""  # wds dataset path
    source_url: str = ""
    image_column_name: str = "jpg"
    cls_column_name: str = "cls"
    max_seq_len: int = 4096
    min_res: Optional[int] = None
    image_decoding_mode: str = "rgb8"
    seed: int = 42
    shuffle_size: int = 5000


def get_dataset(args: DatasetArgs, input_processor: SpihtInputProcessor):
    """
    returns a datset that contains input_ids and metadata_ids
    """

    dataset_type = args.dataset_type
    cls_column_name = args.cls_column_name
    dataset = args.dataset
    image_column_name = args.image_column_name
    max_seq_len = args.max_seq_len
    min_res = args.min_res
    handler = wds.handlers.reraise_exception

    if dataset_type == "wds-image":

        assert args.image_decoding_mode in {"rgb8", "l8", "rgba8"}

        ds = (
            wds.WebDataset(dataset)
            .shuffle(args.shuffle_size, rng=random.Random(args.seed))
            .decode(args.image_decoding_mode, handler=handler)
            .rename(pixel_values=image_column_name, handler=handler)
            .rename(label=cls_column_name, handler=handler)
        )
        if min_res:
            ds = ds.select(FilterMinRes(min_res))

        ds = ds.map(
            _SpihtImagePreprocessor(
                input_processor.spiht_configuration, max_seq_len, None
            )
        )

    elif dataset_type == "wds-preprocessed":
        ds = (
            wds.WebDataset(dataset)
            .decode(handler=handler)
            .rename(encoding_result="encoding_result.pyd", handler=handler)
            .rename(label=cls_column_name, handler=handler)
        )
    else:
        raise ValueError(dataset_type)

    ds = ds.map(_SpihtHtmlFormatter(input_processor, max_seq_len))

    return ds


def resize_to_max(pixel_values, max_res):
    _, h, w = pixel_values.shape
    if max(h, w) > max_res:
        aspect_ratio = h / w
        if h > w:
            h = max_res
            w = int(h / aspect_ratio)
        else:
            w = max_res
            h = int(aspect_ratio * w)

        rz = transforms.Resize(min(h, w), antialias=True)
        pixel_values = rz(pixel_values)
    return pixel_values


class FilterMinRes:
    def __init__(self, min_res: int):
        self.min_res = min_res

    def __call__(self, row):
        h, w = row["pixel_values"].shape[:-1]
        return min(h, w) >= self.min_res
