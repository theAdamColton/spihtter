import os
import unittest
import pywt
import numpy as np
import spiht
from spiht import SpihtSettings

from ..utils import bytes_to_bits

from ..spiht_streaming_decoder import (
    streaming_decode,
    get_llh_llw,
    get_slices_and_arr_h_arr_w,
)
from ..utils import imload, imsave


class TestDecShapes(unittest.TestCase):
    def test_streaming_decode_metadata(self):
        for image_file in os.listdir("./images"):
            print("testing...", image_file)
            im = imload("./images/" + image_file)
            c, h, w = im.shape
            spiht_settings = SpihtSettings()
            level = 8
            encoding_result = spiht.encode_image(
                im, spiht_settings, level=level, max_bits=50_000
            )

            rec_image_0, spiht_metadata_0 = spiht.decode_image(
                encoding_result, spiht_settings, return_metadata=True
            )
            rec_image_0 = spiht.decode_image(encoding_result, spiht_settings)
            bits = bytes_to_bits(encoding_result.encoded_bytes)
            shapes = pywt.wavedecn_shapes(
                (c, h, w),
                wavelet=spiht_settings.wavelet,
                mode=spiht_settings.mode,
                level=level,
                axes=(-2, -1),
            )
            slices, rec_arr_h, rec_arr_w = get_slices_and_arr_h_arr_w(shapes)
            ll_h, ll_w = get_llh_llw(shapes)
            rec_arr = np.zeros((c, rec_arr_h, rec_arr_w), dtype=np.int32)

            spiht_metadata = []
            for row in streaming_decode(
                bits, rec_arr, slices, encoding_result.max_n, ll_h, ll_w, level
            ):
                spiht_metadata.append(row)

            # first, compares the two images of both decoding algorithms
            rec_image = spiht.spiht_wrapper.decode_from_rec_arr(
                rec_arr, h, w, level, spiht_settings, slices
            )

            imsave(
                rec_image_0,
                f"./testout/test_streaming_decoder.test_streaming_decode_metadata.rec_image_0.{image_file}",
            )
            imsave(
                rec_image,
                f"./testout/test_streaming_decoder.test_streaming_decode_metadata.rec_image.{image_file}",
            )

            spiht_metadata = np.array(spiht_metadata, dtype=np.int32)

            length_difference = len(spiht_metadata_0) - len(spiht_metadata)

            # there's a length difference because the rust decoder uses bytes
            # which are in groups of 8 bits
            # So the rust encoder sometimes has up to 8 extra bits
            self.assertLessEqual(length_difference, 8)
            self.assertGreaterEqual(length_difference, 0)

            if length_difference > 0:
                spiht_metadata_0 = spiht_metadata_0[:-length_difference]

            self.assertTrue(
                np.array_equal(
                    spiht_metadata,
                    spiht_metadata_0,
                )
            )
