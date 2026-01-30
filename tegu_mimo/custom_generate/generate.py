from typing import Union, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from transformers.generation.utils import GenerationMixin, GenerateDecoderOnlyOutput, GenerateNonBeamOutput
import logging

logger = logging.getLogger(__name__)


def compute_contrast_logits(expert_logits, amateur_logits, alpha=0.5, beta=0.1):
    # log_softmax + apc
    curr_alpha = alpha
    curr_beta = beta

    expert_log_probs = F.log_softmax(expert_logits, dim=-1)
    amateur_log_probs = F.log_softmax(amateur_logits, dim=-1)

    diff_log_probs = (1 + curr_alpha) * expert_log_probs - curr_alpha * amateur_log_probs

    if curr_beta > 0:
        expert_probs = torch.exp(expert_log_probs)
        expert_max_probs, _ = torch.max(expert_probs, dim=-1, keepdim=True)
        
        mask = expert_probs < (curr_beta * expert_max_probs)
        diff_log_probs = diff_log_probs.masked_fill(mask, -float('inf'))

    return diff_log_probs

def _mtp_decoding(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: "BaseStreamer" = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    """
    MTP Temporal Self-Contrast Decoding implementation.
    """
    mtp_alpha = getattr(generation_config, "mtp_alpha", model_kwargs.pop("mtp_alpha", 0.5))
    
    if getattr(generation_config, "num_beams", 1) != 1:
        raise ValueError("MTP decoding currently only supports num_beams == 1")
    
    pad_token_id = generation_config.pad_token_id
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    if pad_token_id is None:
        pad_token_id = 151643  # for MiMo-7B

    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None

    batch_size, cur_length = input_ids.shape[:2]
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    if model_kwargs.get("cache_position") is None:
        cache_position = torch.arange(cur_length, device=input_ids.device)
        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            if isinstance(cache, tuple):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, "get_seq_length"):
                past_length = cache.get_seq_length()
            
            if past_length > 0:
                cache_position = cache_position[past_length:]

        model_kwargs["cache_position"] = cache_position

    this_peer_finished = False

    cached_hidden_states = None
    is_prefill_step = True
    mtp_past_key_values = None

    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True, 
        )
        expert_logits = outputs.logits[:, -1, :].float()
        
        last_hidden_states = outputs.hidden_states[-1]  # [bs, seq_len, hidden_size]

        if is_prefill_step:
            mtp_model_inputs = {
                "input_ids": model_inputs['input_ids'][:, 1:], 
                "hidden_states": last_hidden_states[:, :-1, :],
                "attention_mask": model_inputs['attention_mask'][:, :-1],
                "position_ids": model_inputs['position_ids'][:, :-1],
                "cache_position": model_inputs['cache_position'][:-1],
            }
            is_prefill_step = False
        else:
            mtp_model_inputs = {
                "input_ids": model_inputs['input_ids'],  
                "hidden_states": cached_hidden_states,  
                "attention_mask": model_inputs['attention_mask'][:, :-1],
                "position_ids": model_inputs['position_ids'] - 1,
                "cache_position": model_inputs['cache_position'] - 1,
            }

        cached_hidden_states = last_hidden_states[:, -1:, :].clone()

        mtp_output = model.get_mtp_logits(
            **mtp_model_inputs,
            past_key_values=mtp_past_key_values,  
            use_cache=True,                   
            logits_to_keep=1
        )
        mtp_past_key_values = mtp_output.past_key_values
        amateur_logits = mtp_output.logits[:, -1, :].float()

        next_token_logits = compute_contrast_logits(expert_logits, amateur_logits, alpha=mtp_alpha)

        next_token_logits = next_token_logits.to(input_ids.device)
        next_token_scores = logits_processor(input_ids, next_token_logits)

        if return_dict_in_generate and output_scores:
            scores += (next_token_scores,)
        if return_dict_in_generate and output_logits:
            raw_logits += (expert_logits,)

        if do_sample:
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        return GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            logits=raw_logits,
            past_key_values=model_kwargs.get("past_key_values"),
        )
    else:
        return input_ids

def generate(model, *args, **kwargs):
    """
    Custom generate function for MTP contrastive decoding.
    """
    mtp_alpha = kwargs.pop("mtp_alpha", 0.5)

    gen_config = kwargs.get("generation_config")
    
    if gen_config is None:
        gen_config = GenerationConfig.from_model_config(model.config)
        kwargs["generation_config"] = gen_config
    
    kwargs["generation_config"].mtp_alpha = mtp_alpha

    generation_outputs = GenerationMixin.generate(
        model, *args, custom_generate=_mtp_decoding, **kwargs
    )
    return generation_outputs