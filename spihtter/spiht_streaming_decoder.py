"""
When generating image tokens, the LlamaSpihtter model expects to recieve as
input information not only from the input ids, but also spiht metadata ids.
Spiht metadata is given concerning the attributes of the NEXT DWT coefficient
to be decoded. So while incrementally decoding Spiht bits, there must also be a
running Spiht algorithm that can update itself with the new tokens that are
generated, and yeild metadata.
"""

from dataclasses import dataclass
from typing import Generator, Iterable, List, Any
import numpy as np
import torch
import pywt

from spiht import SpihtSettings

from .spiht_configuration import BaseSpihtConfiguration
from .spiht_image import SpihtImage


class SpihtStreamingDecoder:
    def __init__(
        self,
        spiht_configuration: BaseSpihtConfiguration,
        spiht_image: SpihtImage,
        bit_stream,
    ):
        c, h, w = spiht_image.shape

        level = spiht_configuration.get_level(h, w)

        shapes = pywt.wavedecn_shapes(
            (1, h, w),
            wavelet=spiht_configuration.wavelet,
            mode=spiht_configuration.mode,
            level=level,
            axes=(-2, -1),
        )

        ll_h, ll_w = get_llh_llw(shapes)
        slices, arr_h, arr_w = get_slices_and_arr_h_arr_w(shapes)

        self.rec_arr = np.zeros((c, arr_h, arr_w), dtype=np.int32)

        self.bit_stream = bit_stream

        self.generator_stream = streaming_decode(
            self.bit_stream,
            self.rec_arr,
            slices,
            spiht_image.max_n,
            ll_h,
            ll_w,
            level,
        )

    def __next__(self):
        return next(self.generator_stream)


class SpihtExhausted(Exception):
    """
    This is raised when the Spiht decoder has emptied all of it's queues and
    has nothing else to do
    """


def get_llh_llw(shapes):
    """
    h: original height of the input image
    w: original width of the input image

    Returns the shape of the top left low filtered coeffs
    """
    *_, ll_h, ll_w = shapes[0]
    return ll_h, ll_w


def get_slices_and_arr_h_arr_w(shapes):
    """
    Returns the same exact slices that would be used in the Wavedec
    same as pywt.coeffs_to_array slices

    Only works for a 3D array, with Wavedec2

    Returns:
    slices, height of rec array, width of rec array
    """
    *_, start_h, start_w = shapes[0]

    slices: List[Any] = [(slice(None), slice(0, start_h), slice(0, start_w))]
    for shape in shapes[1:]:
        shape_ad = shape["ad"]
        shape_da = shape["da"]
        shape_dd = shape["dd"]
        slices.append(
            {
                "ad": (
                    slice(None),
                    slice(0, shape_ad[1]),
                    slice(start_w, start_w + shape_ad[2]),
                ),
                "da": (
                    slice(None),
                    slice(start_h, start_h + shape_da[1]),
                    slice(0, shape_da[2]),
                ),
                "dd": (
                    slice(None),
                    slice(start_h, start_h + shape_dd[1]),
                    slice(start_w, start_w + shape_dd[2]),
                ),
            }
        )

        start_h += shape["dd"][1]
        start_w += shape["dd"][2]

    return slices, start_h, start_w


# SPIHT decoder utils


def has_descendents_past_offspring(i, j, h, w):
    if (2 * i + 1) * 2 + 1 >= h or (2 * j + 1) * 2 + 1 >= w:
        return False
    else:
        return True


def set_bit(x, n, bit):
    sign = x >= 0
    if bit:
        if sign:
            return x | (1 << n)
        else:
            return -((-x) | 1 << n)
    else:
        if sign:
            return x & ~(1 << n)
        else:
            return -((-x) & ~(1 << n))


class EndDecoding(Exception):
    pass


@dataclass
class CoefficientMetadata:
    depth: int
    filter: int
    channel: int
    height: int
    width: int

    @property
    def coords(self):
        return self.channel, self.height, self.width

    def get_filter_of_offspring(self):
        if self.filter == filter_to_index("ll"):
            if self.height % 2 == 1 and self.width % 2 == 1:
                return filter_to_index("dd")
            if self.height % 2 == 0 and self.width % 2 == 1:
                return filter_to_index("ad")
            else:
                return filter_to_index("da")
        else:
            return self.filter

    def get_offspring(self, ll_h, ll_w, rec_arr_h, rec_arr_w):
        if self.height * 2 + 1 >= rec_arr_h or self.width * 2 + 1 >= rec_arr_w:
            return []

        offspring_filter = self.get_filter_of_offspring()

        if self.height < ll_h and self.width < ll_w:
            if self.height % 2 == 0 and self.width % 2 == 0:
                return []
            # index relative to the top left chunk corner
            # can be (0,0), (0,2), (2,0), (2,2)
            sub_child_i, sub_child_j = self.height // 2 * 2, self.width // 2 * 2
            # can be (0,1), (1,0) or (1,1)
            chunk_i, chunk_j = self.height % 2, self.width % 2
            return [
                CoefficientMetadata(
                    self.depth - 1,
                    offspring_filter,
                    self.channel,
                    chunk_i * ll_h + sub_child_i,
                    chunk_j * ll_w + sub_child_j,
                ),
                CoefficientMetadata(
                    self.depth - 1,
                    offspring_filter,
                    self.channel,
                    chunk_i * ll_h + sub_child_i,
                    chunk_j * ll_w + sub_child_j + 1,
                ),
                CoefficientMetadata(
                    self.depth - 1,
                    offspring_filter,
                    self.channel,
                    chunk_i * ll_h + sub_child_i + 1,
                    chunk_j * ll_w + sub_child_j,
                ),
                CoefficientMetadata(
                    self.depth - 1,
                    offspring_filter,
                    self.channel,
                    chunk_i * ll_h + sub_child_i + 1,
                    chunk_j * ll_w + sub_child_j + 1,
                ),
            ]

        return [
            CoefficientMetadata(
                self.depth - 1,
                offspring_filter,
                self.channel,
                2 * self.height,
                2 * self.width,
            ),
            CoefficientMetadata(
                self.depth - 1,
                offspring_filter,
                self.channel,
                2 * self.height,
                2 * self.width + 1,
            ),
            CoefficientMetadata(
                self.depth - 1,
                offspring_filter,
                self.channel,
                2 * self.height + 1,
                2 * self.width,
            ),
            CoefficientMetadata(
                self.depth - 1,
                offspring_filter,
                self.channel,
                2 * self.height + 1,
                2 * self.width + 1,
            ),
        ]

    def get_local_position(self, slices, level):
        if self.depth == level:
            local_h = self.height / slices[0][1].stop
            local_w = self.width / slices[0][2].stop
        else:
            depth_i = level - self.depth
            filter_i = self.filter
            filter_slice = slices[depth_i][index_to_filter(filter_i)]
            local_h = (self.height - filter_slice[1].start) / (
                filter_slice[1].stop - filter_slice[1].start
            )
            local_w = (self.width - filter_slice[2].start) / (
                filter_slice[2].stop - filter_slice[2].start
            )

        # as an integer from -100_000 to 100_000
        return local_h * 200_000.0 - 100_000.0, local_w * 200_000.0 - 100_000.0


def filter_to_index(filter):
    return {"ll": 0, "da": 1, "ad": 2, "dd": 3}[filter]


def index_to_filter(x):
    match x:
        case 0:
            return "ll"
        case 1:
            return "da"
        case 2:
            return "ad"
        case 3:
            return "dd"
    raise ValueError(x)


def streaming_decode(
    spiht_bits,
    rec_arr: np.ndarray,
    slices,
    n: int,
    ll_h: int,
    ll_w: int,
    level: int,
):
    """
    Args:
        spiht_bits: Iterable over the encoded bits created by the spiht encoder
        h: height of the original image
        w: width of the original image
        c: number of channels in the encoded coefficients array
        n: The starting log2 value for the spiht decoder algorithm, same as the
            'n' variable in the original spiht paper
        spiht_settings: the settings used for the DWT wavedec2 used to encode the array.
        level: level used to encode to the spiht bits

    Returns a generator.
    Yeilds a stream of:
        bit: The current bit
        rec_arr: A reference to the current decoded array

        Spiht metadata, which is a 8 length torch.LongTensor vector
        This contains the following:
        a: action ID, from 0 to 6 (inclusive)
        h: next coeff height as a relative position in the filter
        w: next coeff width as a relative position in the filter
        c: next coeff channel
        f: next coeff filter, as an integer from 0 to 3 (inclusive)
            0 being the 'll' top level, and then in order: H, V, D
            'll', 'da', 'ad', 'dd' is mapped to: 0, 1, 2, 3
        d: next coeff filter depth
        n: next coeff n value (the variable n)
        x: next value of the coefficient in the rec_arr

    Position/action information:
        Position/action information is given of the NEXT coefficient to have information
        outputted
        This is the position relative to the unpacked coefficients array. For example,
        the very bottom right coefficient will only have a maximum position of h//2
        and w//2, because this is the relative position inside of the 'dd' coeffs.

        Heights and widths are given as relative positions, for example, a
            pixel from the center of a filter array will be returned as 0,0
    """
    c, arr_h, arr_w = rec_arr.shape

    spiht_bits_iterator = iter(spiht_bits)

    Array = lambda x: torch.LongTensor(x)

    def pop():
        try:
            bit = next(spiht_bits_iterator)
        except StopIteration:
            raise EndDecoding()

        return bit

    lis = []
    for i in range(ll_h):
        for j in range(ll_w):
            if i % 2 == 0 and j % 2 == 0:
                continue
            for k in range(c):
                # 0th is type, 1st is the coefficient metadata
                lis.append(
                    ("A", CoefficientMetadata(level, filter_to_index("ll"), k, i, j))
                )
    lip = []
    for i in range(ll_h):
        for j in range(ll_w):
            for k in range(c):
                lip.append(
                    (
                        CoefficientMetadata(
                            level,
                            filter_to_index("ll"),
                            k,
                            i,
                            j,
                        )
                    )
                )
    lsp = []

    try:
        while n >= 0:
            # sorting pass
            # stores the lsp len at the beginning of this n iteration
            lsp_len = len(lsp)

            new_lip = []
            for coefficient in lip:
                # action 0
                yield Array(
                    (
                        0,  # action id
                        *coefficient.get_local_position(slices, level),  # height, width
                        coefficient.channel,  # channel_id
                        coefficient.filter,  # filter_id
                        coefficient.depth,  # depth_id
                        n,  # n_id
                        rec_arr[coefficient.coords],  # x
                    )
                )
                is_element_sig = pop()

                if is_element_sig:
                    # action 1
                    yield Array(
                        (
                            1,  # action id
                            *coefficient.get_local_position(
                                slices, level
                            ),  # height, width
                            coefficient.channel,  # channel_id
                            coefficient.filter,  # filter_id
                            coefficient.depth,  # depth_id
                            n,  # n_id
                            rec_arr[coefficient.coords],  # x
                        )
                    )
                    sign = pop()

                    # 1 or -1
                    sign = sign * 2 - 1

                    rec_arr[coefficient.coords] = 1.5 * 2**n * sign
                    lsp.append(coefficient)
                else:
                    new_lip.append(coefficient)

            lip = new_lip

            lis_retain = []

            while len(lis) > 0:
                _type, coefficient = lis.pop(0)

                if _type == "A":
                    # action 2
                    yield Array(
                        (
                            2,  # action id
                            *coefficient.get_local_position(
                                slices, level
                            ),  # height, width
                            coefficient.channel,  # channel_id
                            coefficient.filter,  # filter_id
                            coefficient.depth,  # depth_id
                            n,  # n_id
                            rec_arr[coefficient.coords],  # x
                        )
                    )
                    is_set_sig = pop()

                    if is_set_sig:
                        # processes the four offspring
                        for offspring_coeff in coefficient.get_offspring(
                            ll_h, ll_w, arr_h, arr_w
                        ):
                            # action 3
                            yield Array(
                                (
                                    3,
                                    *offspring_coeff.get_local_position(
                                        slices, level
                                    ),  # height, width
                                    offspring_coeff.channel,  # channel_id
                                    offspring_coeff.filter,  # filter_id
                                    offspring_coeff.depth,  # depth_id
                                    n,  # n_id
                                    rec_arr[offspring_coeff.coords],  # x
                                )
                            )

                            is_element_sig = pop()

                            if is_element_sig:
                                lsp.append(offspring_coeff)
                                # action 4
                                yield Array(
                                    (
                                        4,  # action id
                                        *offspring_coeff.get_local_position(
                                            slices, level
                                        ),  # height, width
                                        offspring_coeff.channel,  # channel_id
                                        offspring_coeff.filter,  # filter_id
                                        offspring_coeff.depth,  # depth_id
                                        n,  # n_id
                                        rec_arr[offspring_coeff.coords],  # x
                                    )
                                )
                                sign = pop()

                                # either 1 or -1
                                sign = sign * 2 - 1

                                rec_arr[offspring_coeff.coords] = 1.5 * 2**n * sign
                            else:
                                lip.append(offspring_coeff)

                        l_exists = has_descendents_past_offspring(
                            coefficient.height, coefficient.width, arr_h, arr_w
                        )
                        if l_exists:
                            lis.append(("B", coefficient))
                    else:
                        # keep lis_element in the lis
                        lis_retain.append((_type, coefficient))

                else:
                    # type B

                    # action 5
                    yield Array(
                        (
                            5,  # action id
                            *coefficient.get_local_position(
                                slices, level
                            ),  # height, width
                            coefficient.channel,  # channel_id
                            coefficient.filter,  # filter_id
                            coefficient.depth,  # depth_id
                            n,  # n_id
                            rec_arr[coefficient.coords],  # x
                        )
                    )
                    is_l_significant = pop()

                    if is_l_significant:
                        for offspring_coeff in coefficient.get_offspring(
                            ll_h, ll_w, arr_h, arr_w
                        ):
                            lis.append(("A", offspring_coeff))
                    else:
                        # keep lis_element in the lis
                        lis_retain.append((_type, coefficient))

            lis = lis_retain

            # refinement pass
            for lsp_i in range(lsp_len):
                coefficient = lsp[lsp_i]

                # action 6
                yield Array(
                    (
                        6,  # action id
                        *coefficient.get_local_position(slices, level),  # height, width
                        coefficient.channel,  # channel_id
                        coefficient.filter,  # filter_id
                        coefficient.depth,  # depth_id
                        n,  # n_id
                        rec_arr[coefficient.coords],  # x
                    )
                )
                bit = pop()

                rec_arr[coefficient.coords] = set_bit(
                    rec_arr[coefficient.coords], n, bit
                )

            n -= 1

    except EndDecoding:
        pass
