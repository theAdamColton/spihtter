import torch
from diffusers import AutoencoderKL
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from spihtter.dataset import get_dataset
from spihtter.spiht_configuration import get_configuration
from spihtter.process_inputs import SpihtInputProcessor, InputProcessorCache
from spihtter.tokenizer import get_simple_tokenizer
from spihtter.utils import imsave


def main(
        spiht_config_name: str = "MnistSpihtConfiguration",
        output_dir:str="out",
        dataset_type:str="hf-image", # or 'wds-image' or 'wds-preprocessed'
        dataset_path:str="mnist", # hf dataset path or wds dataset path
        image_column_name:str = "image",
        max_samples: int = 10,
        max_seq_len:int=512,
        use_vae: bool = False,
        vae_path: str = "stabilityai/sd-vae-ft-mse",  # or "madebyollin/sdxl-vae-fp16-fix"
        device: str = "cpu",
        dtype: str = "float32",  # or float16
        ):

    spiht_config = get_configuration(spiht_config_name)

    # this can be any compatible tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("mistralai/Mistral-7B-v0.1")

    input_processor = SpihtInputProcessor(tokenizer, spiht_config)

    dataset = get_dataset(
        dataset_type=dataset_type,
        dataset=dataset_path,
        image_column_name = image_column_name,
        input_processor =input_processor,
        max_seq_len =max_seq_len,
        max_size = max_samples,
    )

    torch_dtype = getattr(torch, dtype)
    if use_vae:
        vae = (
            AutoencoderKL.from_pretrained(vae_path)
            .to(torch_dtype)
            .to(device)
        )

    for i,row in enumerate(dataset):
        input_ids = row['input_ids']
        spiht_metadata_ids = row['spiht_metadata_ids']

        # take out the start token
        text = tokenizer.decode(input_ids[1:])
        print(repr(text)[:200])
        cache = InputProcessorCache()
        rec_metadata_ids, cache = input_processor.process_input_ids_with_cache(
            input_ids, cache, output_input_processor_cache=True
        )
        image = cache.get_last_image()

        if use_vae:
            with torch.inference_mode():
                image = (
                    vae.decode(
                        torch.from_numpy(image)
                        .to(torch_dtype)
                        .to(device)
                        .unsqueeze(0)
                    )
                    .sample.squeeze(0)
                    .cpu()
                    .numpy()
                )

        imsave(image, f"{output_dir}/dataset sample {i:04}.png")

        if i+2 >= max_samples:
            break


if __name__ == "__main__":
    import jsonargparse
    dataset = jsonargparse.CLI(main)
