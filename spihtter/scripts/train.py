from dataclasses import dataclass
import os
from typing import Optional
from transformers import (
    HfArgumentParser,
    LlamaConfig,
    Trainer,
    TrainingArguments,
)

from spihtter.configuration_mamba import MambaConfig
from spihtter.modelling_llama_spihtter import LlamaSpihtter
from spihtter.modelling_mamba_spihtter import MambaSpihtter
from spihtter.process_inputs import InputProcessorCache, SpihtInputProcessor
from spihtter.spiht_configuration import (
    CifarSpihtConfiguration,
    MnistSpihtConfiguration,
    TinyImagenetSpihtConfiguration,
    BaseSpihtConfiguration
)
from spihtter.tokenizer import get_simple_tokenizer
from spihtter.generation_utils import GenerateImageCallback
from spihtter.utils import imsave
from ..dataset import get_dataset, get_hf_image_dataset



class AutoRegressiveDecoderTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["input_ids"]
        inputs["labels"] = labels
        return super().compute_loss(model, inputs, return_outputs)


@dataclass
class MiscArgs:
    generate_steps: int = 50
    max_bits: int = 256
    model_conf: str = "./model_configurations/llama_tiny.json"
    generation_device: Optional[str] = None
    dataset: str = "mnist"
    dataset_type: str = 'hf-image' # or 'wds-image' or 'wds-preprocessed'
    image_column_name: str = "image"


if __name__ == "__main__":
    train_args, misc_args = HfArgumentParser(
        (TrainingArguments, MiscArgs)
    ).parse_args_into_dataclasses()
    train_args: TrainingArguments
    misc_args: MiscArgs
    max_bits = misc_args.max_bits

    if misc_args.dataset == "mnist":
        spiht_config = MnistSpihtConfiguration()
    elif misc_args.dataset.startswith("cifar"):
        spiht_config = CifarSpihtConfiguration()
    elif "tiny-imagenet" in misc_args.dataset:
        spiht_config = TinyImagenetSpihtConfiguration()
    else:
        spiht_config = BaseSpihtConfiguration()

    tokenizer = get_simple_tokenizer()

    input_processor = SpihtInputProcessor(tokenizer, spiht_config)

    dataset = get_dataset(
        dataset=misc_args.dataset, image_column_name=misc_args.image_column_name, max_seq_len=max_bits, input_processor=input_processor, dataset_type=misc_args.dataset_type
    )

    os.makedirs(train_args.output_dir, exist_ok=True)

    # initializes image prompts
    if misc_args.dataset == "mnist":
        image_prompts = [f"{i}<spiht n=111>" for i in range(10)]
    else:
        # initializes image prompts from the start of some dataset items
        image_prompts = []
        for i, row in enumerate(dataset):
            input_ids = row["input_ids"]

            metdata_ids = row["spiht_metadata_ids"]
            # take out the start token
            text = tokenizer.decode(input_ids[1:])

            close_tag_index = text.index(">")
            image_prompt = text[: close_tag_index + 1]
            image_prompts.append(image_prompt)

            print(repr(image_prompt))

            cache = InputProcessorCache()
            _, cache = input_processor.process_input_ids_with_cache(
                input_ids, cache, output_input_processor_cache=True
            )
            image = cache.get_last_image()

            imsave(image, f"{train_args.output_dir}/dataset sample {i:04}.png")

            if i + 1 >= 8:
                break

    # tokenizes image prompts
    tokenizer.padding_side = "left"
    generation_inputs = tokenizer(
        image_prompts,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
        padding=True,
    )
    tokenizer.padding_side = "right"
    generation_inputs.pop("token_type_ids")
    generation_inputs.update(
        dict(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
    )

    if "llama" in misc_args.model_conf:
        model_conf = LlamaConfig.from_json_file(misc_args.model_conf)
        model = LlamaSpihtter(model_conf)
    elif "mamba" in misc_args.model_conf:
        model_conf = MambaConfig.from_json_file(misc_args.model_conf)
        model = MambaSpihtter(model_conf)
    else:
        raise ValueError(misc_args.model_conf)

    print(
        f"trainable parameters: {model.num_parameters(only_trainable=True) / 1_000_000:.2f} million"
    )

    generation_callback = GenerateImageCallback(
        run_every_steps=misc_args.generate_steps,
        max_length=max_bits,
        spiht_input_processor=input_processor,
        generation_device=misc_args.generation_device,
        output_dir=train_args.output_dir,
        **generation_inputs,
    )

    trainer = AutoRegressiveDecoderTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        callbacks=[generation_callback],
        tokenizer=tokenizer,
    )
    trainer.train(
        resume_from_checkpoint=train_args.resume_from_checkpoint,
    )

    model.save_pretrained(f"{train_args.output_dir}/most_recent/")
