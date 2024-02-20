import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import unicodedata
import re

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

def pad_truncate_to(x, n, fill=0):
    # x: Shape S ...
    s, *rest = x.shape
    if s < n:
        # pads
        out = torch.full((n, *rest), fill, dtype=x.dtype, device=x.device)
        out[:s] = x
        x = out
    elif s > n:
        # truncates
        x = x[:n]
    return x


def bits_to_int(bitlist):
    """
    little endian
    """
    out = 0
    for i, bit in enumerate(bitlist):
        out = out | (int(bit) << i)
    return out


def int_to_bits(x):
    """
    little endian
    """
    return format(x, "b")[::-1]


def bytes_to_bits(bytes: bytes):
    np_bytes = np.frombuffer(bytes, np.uint8)
    np_bits = np.unpackbits(np_bytes, bitorder="little")
    return np_bits


def bits_to_bytes(bits: np.ndarray):
    return np.packbits(bits, bitorder="little").tobytes()


def imload(path) -> np.ndarray:
    im = np.asarray(Image.open(path))
    return np.moveaxis(im, -1, 0) / 255


def imsave(im: np.ndarray, dest):
    if im.ndim == 3:
        c = im.shape[0]
        if c == 1:
            im = im[0]
        else:
            im = np.moveaxis(im, 0, -1)
    im = im.clip(0.0, 0.9999999) * 255
    im = im.astype(np.uint8)
    pil_im = Image.fromarray(im)
    print("saving im to ", dest)
    return pil_im.save(dest)


def imshow(x, ax=None):
    if isinstance(x, torch.Tensor):
        x = x.cpu().float()
        x = x.clamp(0.0, 1.0)

        if len(x.shape) > 2:
            x = x.permute(1, 2, 0)
    elif isinstance(x, np.ndarray):
        x = np.moveaxis(x, 0, -1)
        x = np.clip(x, 0, 1)

    if ax is None:
        ax = plt
        ax.imshow(x)
        ax.axis("off")
        plt.show()
    else:
        ax.axis("off")
        ax.tick_params(
            axis="both",
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )
        ax.imshow(x)
