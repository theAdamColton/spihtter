import unittest

import random
import torch
from transformers import LlamaConfig

from spihtter.process_inputs import SpihtInputProcessor
from spihtter.spiht_configuration import SpihtConfiguration
from ..modelling_llama_spihtter import LlamaSpihtter
from ..tokenizer import get_simple_tokenizer


class Tests(unittest.TestCase):
    def _get_setup(self):
        tk = get_simple_tokenizer(["0123456789"])
        config = LlamaConfig.from_json_file("./model_configurations/llama_tiny.json")
        model = LlamaSpihtter(config)
        spiht_config = SpihtConfiguration()
        input_processor = SpihtInputProcessor(tk, spiht_config)
        return tk, model, input_processor, spiht_config

    def test_generation_llama(self):
        seed = 42
        rng = random.Random(seed)
        tk, model, input_processor, spiht_config = self._get_setup()
        tokenizer_outputs = tk(
            "123141148 <spiht h=11 f=12 n=14 w=10>", return_tensors="pt"
        )

        for _ in range(30):
            torch.manual_seed(rng.randbytes(1)[0])
            outputs = model.generate(
                tokenizer_outputs["input_ids"],
                spiht_input_processor=input_processor,
                past_input_processor_cache=[],
                do_sample=True,
                temperature=10.0,
                max_length=100,
            )
