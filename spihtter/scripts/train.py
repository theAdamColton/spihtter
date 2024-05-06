import torch
from transformers import (
    Trainer,
    TrainingArguments,
)
import hydra
import hydra
from transformers.training_args import AcceleratorConfig

from .config import LaunchConfig
from spihtter.model_factory import get_model
from spihtter.process_inputs import SpihtInputProcessor
from spihtter.spiht_configuration import SpihtConfiguration
from spihtter.tokenizer import get_simple_tokenizer
from spihtter.generation_utils import GenerateImageCallback
from spihtter.dataset import get_dataset


class AutoRegressiveDecoderTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["input_ids"]
        inputs["labels"] = labels
        return super().compute_loss(model, inputs, return_outputs)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(config: LaunchConfig):
    spiht_config = SpihtConfiguration(**config.spiht)
    model_config = config.model

    tokenizer = get_simple_tokenizer()

    input_processor = SpihtInputProcessor(tokenizer, spiht_config)

    dataset = get_dataset(config.data, input_processor)

    image_prompts = list(config.generation.prompts)

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

    generation_callback = GenerateImageCallback(
        config.generation, input_processor, generation_inputs
    )

    model = get_model(model_config)

    model = model.to(torch.float16)

    # gets rid of the keys that start with _
    train_args = {k: v for k, v in config.train.items() if not k.startswith("_")}
    train_args["accelerator_config"] = AcceleratorConfig(
        **train_args["accelerator_config"]
    )

    train_args = TrainingArguments(**train_args)

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


if __name__ == "__main__":
    main()
