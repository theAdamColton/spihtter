from dataclasses import asdict
from typing import Optional
import torch
import numpy as np
from torch import nn
from torchvision import transforms
import webdataset as wds
import datasets
import spiht

from spihtter.process_inputs import InputProcessorCache, SpihtInputProcessor
from spihtter.spiht_configuration import BaseSpihtConfiguration
from spihtter.spiht_image import SpihtImage
from spihtter.utils import pad_truncate_to



def _to_torch_rgb(pixel_values):
    if isinstance(pixel_values, np.ndarray):
        pixel_values = torch.from_numpy(pixel_values)

    if pixel_values.dtype == torch.uint8:
        pixel_values = pixel_values / 255

    if pixel_values.ndim == 2:
        pixel_values = pixel_values[None, :, :]

    assert isinstance(pixel_values, torch.Tensor)

    return pixel_values

def _to_np_rgb(pixel_values):
    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values.numpy()
    if pixel_values.dtype == np.uint8:
        pixel_values = pixel_values / 255
    if pixel_values.ndim == 2:
        pixel_values = pixel_values[None, :, :]
    assert isinstance(pixel_values, np.ndarray)
    return pixel_values


class _SpihtVaePreprocessor:
    def __init__(self, spiht_configuration:BaseSpihtConfiguration, max_seq_len: Optional[int] = None, bpp: Optional[float] = None, vae=None):
        self.spiht_configuration = spiht_configuration
        self.vae = vae
        self._spiht_image_preprocessor = _SpihtImagePreprocessor(spiht_configuration, max_seq_len, bpp)

    def __call__(self, row: dict):
        pixel_values = row.pop('pixel_values')
        pixel_values = _to_torch_rgb(pixel_values)
        with torch.inference_mode():
            z = self.vae.encode(
                pixel_values.to(self.vae.dtype).to(self.vae.device).unsqueeze(0)
            ).latent_dist.mean[0]
        z = z.cpu().to(torch.float32).numpy()
        return self._spiht_image_preprocessor(dict(pixel_values = z,**row)
                )
        


class _SpihtImagePreprocessor:
    def __init__(self, spiht_configuration:BaseSpihtConfiguration, max_seq_len:Optional[int], bpp: Optional[float]):
        self.spiht_configuration = spiht_configuration
        self.max_seq_len = max_seq_len
        self.bpp = bpp
        assert max_seq_len or bpp

    def __call__(self, row:dict):
        conf = self.spiht_configuration
        max_seq_len = self.max_seq_len

        pixel_values = row.pop('pixel_values')

        pixel_values = _to_np_rgb(pixel_values)

        c, h, w = pixel_values.shape

        assert c == self.spiht_configuration.image_channels, f"{c} != {self.spiht_configuration.image_channels}"

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
            encoding_result = row['encoding_result']
            assert isinstance(encoding_result, spiht.EncodingResult)
        else:
            encoding_result = spiht.EncodingResult.from_dict(row)
        spiht_image = SpihtImage.from_encoding_result(encoding_result, conf)

        label = row.pop('label')
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

def get_dataset(
        dataset_type="hf-image", # or 'wds-image' or 'wds-preprocessed'
        dataset="mnist", # hf dataset path or wds dataset path
        image_column_name = "image",
        input_processor: SpihtInputProcessor=None,
        max_seq_len: int=None,
        max_size: int = None,
        ):
    """
    returns a datset that contains input_ids and metadata_ids
    """
    if dataset_type == "hf-image":
        ds = get_hf_image_dataset(dataset, image_column_name)
        if max_size:
            ds = ds.select(range(max_size))
    elif dataset_type == "wds-image":
        ds = get_wds_image_dataset(dataset, image_column_name=image_column_name)
    elif dataset_type == "wds-preprocessed":
        ds = get_wds_preprocessed_dataset(dataset)
    else:
        raise ValueError(dataset_type)

    if 'image' in dataset_type:
        ds = ds.map(_SpihtImagePreprocessor(input_processor.spiht_configuration, max_seq_len, None))
        if isinstance(ds, datasets.Dataset):
            ds.set_format(None)

    ds = ds.map(_SpihtHtmlFormatter(input_processor, max_seq_len))
    if isinstance(ds, datasets.Dataset):
        ds.set_format('torch')

    return ds
    

def get_hf_image_dataset(
    dataset, image_column_name="image"
):
    ds = (
        datasets.load_dataset(
            dataset,
            split="train",
        )
        .rename_column(image_column_name, "pixel_values")
    )
    ds.set_format('torch', columns=['pixel_values'], output_all_columns=True)
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


class ResizeToMax(nn.Module):
    def __init__(self, max_res):
        self.max_res = max_res

    def __call__(self, pixel_values):
        pixel_values = resize_to_max(pixel_values, self.max_res)
        return pixel_values

class FilterMinRes:
    def __init__(self, min_res):
        self.min_res = min_res

    def __call__(self,row):
        if self.min_res is None:
            return True
        return min(row['pixel_values'].shape[1:]) >= self.min_res

def get_wds_image_dataset(
    path, max_res=1024, min_res=None, handler=wds.handlers.reraise_exception, image_column_name="jpg",
):
    return (
        wds.WebDataset(path)
        .decode("torchrgb8", handler=handler)
        .rename(pixel_values=image_column_name, handler=handler)
        .rename(label="cls")
        .select(FilterMinRes(min_res))
        .map_dict(pixel_values=ResizeToMax(max_res), handler=handler)
    )

def get_wds_preprocessed_dataset(path):
    return (
            wds.WebDataset(path)
            .decode()
            .rename(encoding_result='encoding_result.pyd')
            .rename(label="cls")
        )
