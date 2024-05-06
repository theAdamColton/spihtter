import omegaconf
from dataclasses import dataclass, field
import os
from typing import Any
from transformers import (
    Trainer,
    TrainingArguments,
)
import hydra
import hydra
from hydra.core.config_store import ConfigStore
from transformers.training_args import AcceleratorConfig

from spihtter.model_factory import get_model
from spihtter.process_inputs import SpihtInputProcessor
from spihtter.spiht_configuration import SpihtConfiguration
from spihtter.tokenizer import get_simple_tokenizer
from spihtter.generation_utils import GenerateImageCallback, GenerateImageConfig
from spihtter.dataset import DatasetArgs, get_dataset


class AutoRegressiveDecoderTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["input_ids"]
        inputs["labels"] = labels
        return super().compute_loss(model, inputs, return_outputs)


@dataclass
class SpihtterTrainingArguments(TrainingArguments):
    """overrides some fields to make compatible with omegaconf"""

    output_dir: str = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    lr_scheduler_kwargs: Any = None
    debug: Any = None
    fsdp: Any = ""
    fsdp_config: Any = None
    accelerator_config: Any = None  # = AcceleratorConfig()
    dispatch_batches: Any = False
    deepspeed: Any = None
    report_to: Any = None
    gradient_checkpointing_kwargs: Any = None
    optim_target_modules: Any = None


@dataclass
class LaunchConfig:
    train: SpihtterTrainingArguments = field(default_factory=SpihtterTrainingArguments)
    data: DatasetArgs = field(default_factory=DatasetArgs)
    generation: GenerateImageConfig = field(default_factory=GenerateImageConfig)
    spiht: SpihtConfiguration = field(default_factory=SpihtConfiguration)
    model: Any = None


cs = ConfigStore.instance()
cs.store(name="base_config", node=LaunchConfig)


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
