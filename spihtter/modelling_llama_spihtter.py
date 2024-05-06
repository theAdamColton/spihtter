"""
LlamaSpihtter is a wrapper around LlamaForCausalLM. 

LlamaSpihtter is the same as LlamaForCausalLM, but uses token embeddings which
are computed from both the input ids and the spiht embedder.
"""

import torch
from typing import Optional, List
import torch
from torch import nn
from transformers import LlamaForCausalLM, PreTrainedModel


from .spiht_embedder import SpihtEmbedder
from .generation_utils import SpihtGenerationMixin


class LlamaSpihtter(SpihtGenerationMixin, PreTrainedModel):
    model_type = "llamaspihtter"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaForCausalLM(config)
        self.spihtter_embedder = SpihtEmbedder(dim=config.hidden_size)
        self.post_init()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        spiht_metadata_ids: torch.LongTensor = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.model.model.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds + self.spihtter_embedder(spiht_metadata_ids)

        return self.model.forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=return_dict,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        spiht_input_processor=None,
        past_input_processor_cache=None,
        **kwargs
    ):
        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids,
            past_key_values,
            attention_mask,
            inputs_embeds,
            cache_position,
            **kwargs
        )
        spiht_model_inputs = super().prepare_inputs_for_generation(
            input_ids=model_inputs.pop("input_ids"),
            spiht_input_processor=spiht_input_processor,
            past_input_processor_cache=past_input_processor_cache,
            **kwargs
        )

        model_inputs.update(spiht_model_inputs)

        return model_inputs
