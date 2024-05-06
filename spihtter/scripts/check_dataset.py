import os
from transformers import PreTrainedTokenizerFast
import hydra

from spihtter.dataset import get_dataset
from spihtter.process_inputs import SpihtInputProcessor, InputProcessorCache
from spihtter.spiht_configuration import SpihtConfiguration
from spihtter.utils import imsave
from .config import LaunchConfig


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(config: LaunchConfig):
    output_dir = config.train.output_dir
    os.makedirs(output_dir, exist_ok=True)
    spiht_config = SpihtConfiguration(**config.spiht)

    # this can be any compatible tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct"
    )

    input_processor = SpihtInputProcessor(tokenizer, spiht_config)

    dataset = get_dataset(config.data, input_processor)

    for i, row in enumerate(dataset):
        input_ids = row["input_ids"]
        spiht_metadata_ids = row["spiht_metadata_ids"]

        # take out the start token
        text = tokenizer.decode(input_ids[1:])
        print(repr(text)[:200])
        cache = InputProcessorCache()
        rec_metadata_ids, cache = input_processor.process_input_ids_with_cache(
            input_ids, cache, output_input_processor_cache=True
        )
        image = cache.get_last_image()

        imsave(image, f"{output_dir}/dataset sample {i:04}.png")

        if i + 2 >= config.check_dataset.max_samples:
            break


if __name__ == "__main__":
    main()
