from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import Any
import hydra
from hydra.core.config_store import ConfigStore

from ..dataset import DatasetArgs
from ..generation_utils import GenerateImageConfig
from ..spiht_configuration import SpihtConfiguration


@dataclass
class SpihtterTrainingArguments(TrainingArguments):
    """overrides some fields to make compatible with omegaconf"""

    output_dir: str = "out/"
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
    check_dataset: Any = None


cs = ConfigStore.instance()
cs.store(name="base_config", node=LaunchConfig)
