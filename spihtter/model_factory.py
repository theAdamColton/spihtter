from transformers import LlamaConfig

from .modelling_llama_spihtter import LlamaSpihtter
from .modelling_mamba_spihtter import MambaSpihtter
from .configuration_mamba import MambaConfig


def get_model(config):
    if config.model_type == "llama":
        model_conf = LlamaConfig(**config)
        model = LlamaSpihtter(model_conf)
    elif config.model_type == "mamba":
        model_conf = MambaConfig(**config)
        model = MambaSpihtter(model_conf)
    else:
        raise ValueError(config)

    print(
        f"trainable parameters: {model.num_parameters(only_trainable=True) / 1_000_000:.2f} million"
    )

    return model
