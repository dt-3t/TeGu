from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (Qwen2Attention,
                                                      Qwen2ForCausalLM,
                                                      Qwen2MLP, Qwen2Model,
                                                      Qwen2RMSNorm)

from .configuration_mimo import MiMoConfig

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


class MiMoMTPLayers(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.token_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.final_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config, layer_idx=0)
        self.mlp = Qwen2MLP(config)

    def forward(self, input_embeds,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values: Optional[Cache]=None,
                    output_attentions: Optional[bool]=False,
                    use_cache: Optional[bool]=False,
                    position_embedding: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    cache_position=None,
                    **kwargs):
        input_embeds = self.token_layernorm(input_embeds)
        previous_hidden_states = self.hidden_layernorm(hidden_states)
        hidden_states = self.input_proj(torch.cat([previous_hidden_states, input_embeds], dim=-1))
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states,
                                       attention_mask=attention_mask,
                                       position_ids=position_ids,
                                       past_key_values=past_key_values,
                                       output_attentions=output_attentions,
                                       use_cache=use_cache,
                                       cache_position=cache_position,
                                       position_embedding=position_embedding,
                                       **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class MiMoModel(Qwen2Model):
    config_class = MiMoConfig

    def __init__(self, config: MiMoConfig):
        super().__init__(config)
        self.mtp_layers = nn.ModuleList([MiMoMTPLayers(config) for _ in range(config.num_nextn_predict_layers)])

    def get_mtp_hidden_states(
        self,
        input_ids,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": hidden_states,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                raise NotImplementedError("Sliding window attention is not implemented in MiMo yet.")

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.mtp_layers:
            hidden_states = decoder_layer(
                input_embeds=inputs_embeds,
                hidden_states=hidden_states,
                attention_mask=causal_mask_mapping["full_attention"],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

class MiMoForCausalLM(Qwen2ForCausalLM):
    config_class = MiMoConfig
    def __init__(self, config: MiMoConfig):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = MiMoModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_mtp_logits(
        self,
        input_ids,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        # Definition: hidden_states[t] = BaseModel(input_ids[:t+1])
        # In NTP: logits[t] = lm_head(hidden_states[t-1]), input_ids[t] = sample(logits[t])
        # In this MTP: predicting logits[t+1] requires hidden_states[t-1] and input_ids[t]
        outputs = self.model.get_mtp_hidden_states(
            input_ids=input_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


