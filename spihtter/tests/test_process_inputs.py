import numpy as np
import os
import spiht
import torch
import unittest

import spiht
from spihtter.spiht_image import SpihtImage

from ..utils import bits_to_bytes, imload, bytes_to_bits, imshow, imsave
from ..spiht_configuration import BaseSpihtConfiguration
from ..tokenizer import get_simple_tokenizer
from ..process_inputs import SpihtInputProcessor

os.makedirs("./testout/", exist_ok=True)


class Tests(unittest.TestCase):
    def tokenize(self, tk, string):
        return tk(string, return_tensors="pt").input_ids[0]

    def test_process_with_cace(self):
        tk, config, input_processor = self._get_setup()
        image = SpihtImage.from_html_opening_tag_attrs(dict(h=10, w=13, n=7), config)
        text = image.to_html_opening_tag()

        input_ids = self.tokenize(tk, text)

        metadata_ids = input_processor.process_input_ids_with_cache(
            input_ids,
        )

        self.assertEqual(len(input_ids), len(metadata_ids))

        n = len(input_ids)

        for i in range(n - 1):
            self.assertTrue((metadata_ids[i] == 0).all())
        self.assertTrue((metadata_ids[n - 1] != 0).any())

    def test_parse_decode_spiht_image(self):
        tk, config, input_processor = self._get_setup()
        image = imload("./images/motorcycle.jpg")
        c, h, w = image.shape
        texts = ["<<<<<<<<<<<<><<<><>", None]
        images = [None, SpihtImage.from_file("./images/motorcycle.jpg", config)]
        input_ids, metadata_ids = input_processor.process_normalized_images_texts(
            images, texts
        )

        past_input_processor_cache = None
        for i in range(len(input_ids)):
            input_id = input_ids[i].unsqueeze(0)
            (
                _,
                past_input_processor_cache,
            ) = input_processor.process_input_ids_with_cache(
                input_id,
                past_input_processor_cache,
                output_input_processor_cache=True,
            )

        rec_image = past_input_processor_cache.get_last_image()

        imsave(
            rec_image,
            "./testout/test_process_inputs.test_parse_decode_spiht_image.motorcycle.jpg",
        )

        self.assertLess(((rec_image - image) ** 2).mean(), 0.05)

    def _get_setup(self):
        tk = get_simple_tokenizer(
            [str(i) for i in range(10)] + ["abcdefghijklmnopqrstuvwxyz"]
        )
        conf = BaseSpihtConfiguration()
        proc = SpihtInputProcessor(tk, conf, "\x00", "\x01")
        return tk, conf, proc
