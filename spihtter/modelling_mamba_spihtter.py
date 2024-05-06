from dataclasses import dataclass
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict
import einx

from transformers import PreTrainedModel, TextStreamer
from transformers.utils import ModelOutput

from spihtter.configuration_mamba_spihtter import MambaSpihtterConfig
from spihtter.process_inputs import SpihtInputProcessor

from .generation_utils import SpihtGenerationMixin
from .spiht_embedder import SpihtEmbedder

try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn
except Exception as e:
    print(
        "Error importing CUDA implementation of mamba_ssm! Resorting to slower python implementation.",
        e,
    )
    from .mamba_ops_reference import mamba_inner_ref as mamba_inner_fn

from .configuration_mamba import MambaConfig


def _shallow_clone(a: List[Tuple[torch.Tensor]]):
    b = []
    for layer in a:
        b.append(tuple(x.clone() for x in layer))
    return b


class Mamba(nn.Module):
    """

    This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
    The major differences are :
    -the convolution is done with torch.nn.Conv1d
    -the selective scan is done in PyTorch

    A sequential version of the selective scan is also available for comparison.

    - A Mamba model is composed of several layers, which are ResidualBlock.
    - A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
    - This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
    First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
    Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
    We then multiply it by silu(z).
    See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

    """

    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList(
            [ResidualBlock(config) for _ in range(config.n_layers)]
        )
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = None

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            if self.gradient_checkpointing:
                x = self._gradient_checkpointing_func(layer, x)
            else:
                x = layer(x)

        return x

    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
        # h : (B, ED, N)
        # inputs: (B, d_conv-1, ED)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        # projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(
            config.d_inner, config.dt_rank + 2 * config.d_state, bias=False
        )

        # projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt bias
        dt = torch.exp(
            torch.rand(config.d_inner)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt)
        )  # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(
            config.d_inner, 1
        )
        self.A_log = nn.Parameter(
            torch.log(A)
        )  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        # I'm not sure what this does
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        # I'm not sure what this does
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*ED)

        xz = rearrange(xz, "B L D -> B D L")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        return mamba_inner_fn(
            xz,
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj.weight,
            self.dt_proj.weight,
            self.out_proj.weight,
            self.out_proj.bias,
            A,
            None,  # input-dependent B
            None,  # input-dependent C
            self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (zeros, zeros).
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """

    def step(self, x, cache):
        # x : (B, 1, D)
        # cache : (h, inputs)
        # h : (B, ED, N)
        # inputs : (B, ED, d_conv-1)

        # y : (B, 1, D)
        # cache : (h, inputs)

        assert einx.matches("B 1 D", x)

        h, inputs = cache

        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, 1, ED), (B, 1, ED)

        x_cache = x  # (B, 1, ED)

        # concat over sequence dim
        x = self.conv1d(einx.rearrange("b s ed, b 1 ed -> b ed (s + 1)", inputs, x))
        # take last sequence-wise value
        x = x[..., self.config.d_conv - 1]  # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)  # (B, D)

        y = einx.rearrange("b ed -> b 1 ed", y)

        output = y * z
        output = self.out_proj(output)  # (B, 1, D)

        # prepare cache for next call
        inputs = einx.rearrange(
            "B S2 ED, B S1 ED -> B (S2+S1) ED", inputs[:, 1:], x_cache
        )  # (B, d_conv-1, ED)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(
            self.A_log.float()
        )  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()
        # TODO remove .float()

        deltaBC = self.x_proj(x)  # (B, dt_rank+2*N)

        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )  # (B, dt_rank), (B, N), (B, N)
        delta = F.softplus(self.dt_proj(delta))  # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, ED, N)

        h = deltaA * h + BX  # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        # todo : pq h.squeeze(1) ??
        return y, h.squeeze(1)


# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )

        return output


"""

Encapsulates a Mamba model as language model. It has an embedding layer, and a LM head which maps the model output to logits.

"""


# adapted from https://github.com/johnma2006/mamba-minimal
def from_pretrained(name: str):
    """
    Returns a model loaded with pretrained weights pulled from HuggingFace.

    Args:
        name: As of now, supports
            * 'state-spaces/mamba-2.8b-slimpj'
            * 'state-spaces/mamba-2.8b'
            * 'state-spaces/mamba-1.4b'
            * 'state-spaces/mamba-790m'
            * 'state-spaces/mamba-370m'
            * 'state-spaces/mamba-130m'

    Returns:
        model: a Mamba model configured with the proper parameters and initialized with the proper weights
    """

    from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
    from transformers.utils.hub import cached_file

    def load_config_hf(model_name):
        resolved_archive_file = cached_file(
            model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False
        )
        return json.load(open(resolved_archive_file))

    def load_state_dict_hf(model_name):
        resolved_archive_file = cached_file(
            model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False
        )
        return torch.load(
            resolved_archive_file, weights_only=True, map_location="cpu", mmap=True
        )

    # copy config data
    config_data = load_config_hf(name)
    config = MambaConfig(
        d_model=config_data["d_model"],
        n_layers=config_data["n_layer"],
        vocab_size=config_data["vocab_size"],
    )

    model = MambaSpihtter(config)

    # copy weights
    state_dict = load_state_dict_hf(name)

    new_state_dict = {}
    for key in state_dict:
        if key == "backbone.embedding.weight" or key == "backbone.norm_f.weight":
            new_key = key.replace("backbone.", "")
        else:
            new_key = key.replace("backbone", "mamba")

        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)

    return model


@dataclass
class MambaSpihtterOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: torch.LongTensor = None
    past_mamba_caches: List[Tuple[Optional[torch.Tensor], torch.Tensor]] = None


class MambaSpihtter(SpihtGenerationMixin, PreTrainedModel):
    config_class = MambaSpihtterConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: MambaSpihtterConfig):
        super().__init__(config)
        self.config = config

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.mamba = Mamba(self.config)
        self.norm_f = RMSNorm(self.config.d_model)

        self.lm_head = nn.Linear(
            self.config.d_model, self.config.vocab_size, bias=False
        )
        self.lm_head.weight = self.embedding.weight
        self.spihtter_embedder = SpihtEmbedder(
            dim=config.d_model,
            max_height=config.max_height,
            max_width=config.max_width,
            dwt_channels=config.image_channels,
        )
        self.gradient_checkpointing = False

        self.post_init()

    def _init_weights(self, module):
        initializer_range = self.config.initializer_range
        n_residuals_per_layer = 1  # TODO Change to 2 if we have MLP

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * self.config.n_layers)

    def forward(
        self,
        input_ids=None,
        spiht_metadata_ids=None,
        past_mamba_caches=None,
        return_dict=None,
        labels=None,
        output_attentions=None,
        attention_mask=None,
        output_hidden_states=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        if past_mamba_caches is not None:
            assert not self.training

            # TODO support cache initialization from a batched input
            # Instead we just step a bunch of times
            for seq_i in range(input_ids.shape[1]):
                input_id = input_ids[:, seq_i].unsqueeze(1)
                spiht_metadata_row = spiht_metadata_ids[:, seq_i].unsqueeze(1)

                logits, past_mamba_caches = self.step(
                    input_ids=input_id,
                    spiht_metadata_ids=spiht_metadata_row,
                    past_mamba_caches=past_mamba_caches,
                    return_dict=False,
                )

                if attention_mask is not None:
                    pass
                    # attention masking just means that this particular
                    # input_id does not make a change to the mamba cache
                    # TODO this doesn't work and causes generation to produce garbage
        #                    attention_mask_row = (
        #                        attention_mask[:, seq_i].unsqueeze(-1).unsqueeze(-1)
        #                    )
        #                    for i in range(len(mamba_caches)):
        #                        mamba_caches[i] = (
        #                            past_mamba_caches[i][0] * (~attention_mask_row)
        #                            + mamba_caches[i][0] * attention_mask_row,
        #                            past_mamba_caches[i][1] * (~attention_mask_row)
        #                            + mamba_caches[i][1] * attention_mask_row,
        #                        )
        #
        #                past_mamba_caches = _shallow_clone(mamba_caches)

        else:
            x = self.embedding(input_ids)

            x = x + self.spihtter_embedder(spiht_metadata_ids)

            x = self.mamba(x)
            x = self.norm_f(x)

            logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if return_dict:
            return MambaSpihtterOutput(
                logits=logits, past_mamba_caches=past_mamba_caches, loss=loss
            )

        return logits, caches, loss

    def step(
        self,
        input_ids=None,
        past_mamba_caches=None,
        spiht_metadata_ids=None,
        return_dict=None,
    ):
        # token : (B, 1)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # logits : (B, 1, vocab_size)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        x = self.embedding(input_ids)

        x = x + self.spihtter_embedder(spiht_metadata_ids)

        x, past_mamba_caches = self.mamba.step(x, past_mamba_caches)
        x = self.norm_f(x)

        logits = self.lm_head(x)

        if not return_dict:
            return logits, past_mamba_caches

        return MambaSpihtterOutput(logits=logits, past_mamba_caches=past_mamba_caches)

    def _init_past_mamba_caches(self, batch_size):
        # caches is a list of cache, one per layer
        # cache is composed of : the hidden state, and the last d_conv-1 inputs
        # the hidden state because the update is like an RNN
        # the last d_conv-1 inputs because they are used in a 1d convolution (usually d_conv=4 so this is not large)
        caches = [
            (
                torch.zeros(
                    batch_size,
                    self.config.d_inner,
                    self.config.d_state,
                    device=self.device,
                ),
                torch.zeros(
                    batch_size,
                    self.config.d_conv - 1,
                    self.config.d_inner,
                    device=self.device,
                ),
            )
            for _ in range(self.config.n_layers)
        ]
        return caches

    def _update_model_kwargs_for_generation(
        self,
        outputs: MambaSpihtterOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        """
        we need to override this fn to provide the past_mamba_caches from the
        model output during generation
        """
        model_kwargs["past_mamba_caches"] = outputs.past_mamba_caches
        return super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, standardize_cache_format
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask=None,
        spiht_input_processor: SpihtInputProcessor = None,
        past_input_processor_cache=None,
        past_mamba_caches=None,
        **kwargs
    ) -> Dict[str, Any]:
        if past_mamba_caches is not None:
            # we are in step mode,
            # which means that we feed the model one token at a time
            # a better way to communicate this would be to associate
            # the position_ids with the past_mamba_caches so we know
            # which token the cache corresponds to
            input_ids = input_ids[:, -1].unsqueeze(-1)
        else:
            # we initialize the caches
            past_mamba_caches = self._init_past_mamba_caches(input_ids.shape[0])

        # call the SpihtGenerationMixin to get the spiht metadata ids
        spiht_model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            spiht_input_processor=spiht_input_processor,
            past_input_processor_cache=past_input_processor_cache,
            **kwargs
        )

        spiht_model_inputs.update(
            dict(
                past_mamba_caches=past_mamba_caches,
                attention_mask=attention_mask,
            )
        )

        return spiht_model_inputs
