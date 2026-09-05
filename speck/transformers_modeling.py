"""Transformers wrapper shipped with exported Speck checkpoints."""

import torch
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_speck import SpeckConfig
from .native_speck import RotaryEmbedding, SequenceState
from .native_speck import SpeckForCausalLM as NativeSpeckForCausalLM
from .padding_speck import validate_right_padding


class SpeckPreTrainedModel(PreTrainedModel):
    config_class = SpeckConfig
    base_model_prefix = "native"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _no_split_modules = ["BlockCore"]
    _supports_cache_class = False
    _supports_static_cache = False
    _is_stateful = True
    _tied_weights_keys = {"native.lm_head.weight": "native.embed_tokens.weight"}

    def _init_weights(self, module):
        if isinstance(module, RotaryEmbedding):
            module.frequency = module.frequency.to(dtype=self.dtype)
            module.reset_frequency()


class SpeckForCausalLM(SpeckPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.native = NativeSpeckForCausalLM(config.architecture_config())
        self.post_init()

    def get_input_embeddings(self):
        return self.native.embed_tokens

    def set_input_embeddings(self, value):
        self.native.embed_tokens = value
        self.native.lm_head.weight = value.weight

    def get_output_embeddings(self):
        return self.native.lm_head

    def set_output_embeddings(self, value):
        self.native.lm_head = value

    def state(self, *args, **kwargs):
        return self.native.state(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        logits_to_keep=0,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if output_attentions or output_hidden_states:
            raise ValueError("Speck does not expose attention weights or per-layer hidden states")
        values = input_ids if input_ids is not None else inputs_embeds
        batch_size, length = values.shape[:2]
        has_padding = validate_right_padding(attention_mask, batch_size, length)
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if has_padding and use_cache:
            raise ValueError("right-padded inputs require use_cache=False")
        if past_key_values is not None and not isinstance(past_key_values, SequenceState):
            raise TypeError("Speck requires its native SequenceState cache")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if use_cache and past_key_values is None:
            past_key_values = self.native.state(
                batch_size=batch_size,
                device=values.device,
                dtype=self.dtype,
            )
        position = past_key_values.position if past_key_values is not None else 0
        expected_positions = torch.arange(position, position + length, device=values.device)
        if position_ids is not None:
            expected = expected_positions.unsqueeze(0).expand(batch_size, -1)
            valid = (
                attention_mask.bool()
                if has_padding
                else torch.ones_like(expected, dtype=torch.bool)
            )
            if position_ids.shape != expected.shape or not torch.equal(
                position_ids[valid], expected[valid]
            ):
                raise ValueError("position_ids does not match the Speck sequence")
        keep = logits_to_keep or (getattr(self.config, "num_logits_to_keep", 0) if use_cache else 0)
        last_token_only = bool(keep == 1 and labels is None)
        logits = self.native(
            tokens=input_ids,
            inputs_embeds=inputs_embeds,
            state=past_key_values if use_cache else None,
            last_token_only=last_token_only,
        )
        if keep > 1 and logits.size(1) > keep:
            logits = logits[:, -keep:]
        loss = None
        if labels is not None:
            shifted_logits = logits[:, :-1].contiguous().float()
            shifted_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
                ignore_index=-100,
            )
        if return_dict is False:
            output = (logits, past_key_values if use_cache else None)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            consumed = past_key_values.position
            input_ids = input_ids[:, consumed:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, consumed:]
            inputs_embeds = None
        values = inputs_embeds if inputs_embeds is not None else input_ids
        return {
            "input_ids": None if inputs_embeds is not None else input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "position_ids": torch.arange(
                past_key_values.position if past_key_values is not None else 0,
                (past_key_values.position if past_key_values is not None else 0) + values.size(1),
                device=values.device,
            )[None].expand(values.size(0), -1),
        }


__all__ = ["SequenceState", "SpeckForCausalLM", "SpeckPreTrainedModel"]
