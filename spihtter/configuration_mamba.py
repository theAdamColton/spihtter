import math
from typing import Union
from transformers import PretrainedConfig


class MambaConfig(PretrainedConfig):
    model_type = "mamba"

    def __init__(
        self,
        vocab_size: int = 32,
        pad_vocab_size_multiple: int = 8,
        d_model: int = 512,
        n_layers: int = 8,
        dt_rank: Union[int, str] = "auto",
        d_state: int = 16,
        expand_factor: int = 2,
        d_conv: int = 4,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor=1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        initializer_range: float = 0.02,
        rescale_prenorm_residual: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.d_model = d_model
        self.n_layers = n_layers
        self.dt_rank = dt_rank
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_inner = expand_factor * d_model  # E*D = ED in comments
        self.d_conv = d_conv
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.bias = bias
        self.conv_bias = conv_bias
        self.initializer_range = initializer_range
        self.rescale_prenorm_residual = rescale_prenorm_residual

        super().__init__(
            **kwargs,
        )
