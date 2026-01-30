from typing import Union, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from transformers.generation.utils import GenerationMixin, GenerateDecoderOnlyOutput, GenerateNonBeamOutput
import logging

logger = logging.getLogger(__name__)

def _cmtp_decoding(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: "BaseStreamer" = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    mtp_alpha = getattr(generation_config, "mtp_alpha", model_kwargs.pop("mtp_alpha", 0.5))
    future_offsets = getattr(generation_config, "future_offsets", model_kwargs.pop("future_offsets", [2]))
    amateur_weights = getattr(generation_config, "amateur_weights", model_kwargs.pop("amateur_weights", None))
    
    if not isinstance(future_offsets, list):
        future_offsets = [int(future_offsets)]
    max_offset = max(future_offsets) if future_offsets else 0
    future_offsets_num = len(future_offsets)
    
    if getattr(generation_config, "num_beams", 1) != 1:
        raise ValueError("CMTP decoding currently only supports num_beams == 1")
    
    pad_token_id = generation_config.pad_token_id
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

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

    is_prefill_step = True
    hidden_states_buffer = [] 

    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True, 
        )
        ntp_logits = outputs.logits[:, -1, :].float()  # [batch, vocab_size]
        
        last_hidden_state = outputs.hidden_states[-1] # [batch, seq_len, dim]

        if is_prefill_step:
            seq_len = last_hidden_state.shape[1]
            start_idx = max(0, seq_len - max_offset)
            for i in range(start_idx, seq_len):
                h = last_hidden_state[:, i:i+1, :]  # [batch, 1, dim]
                hidden_states_buffer.append(h)
            is_prefill_step = False
        else:
            hidden_states_buffer.append(last_hidden_state)

        if len(hidden_states_buffer) > max_offset:
            hidden_states_buffer = hidden_states_buffer[-max_offset:]
        
        mtp_logits_list = []
        
        for offset in future_offsets:
            past_hidden = hidden_states_buffer[-offset]
            future_logits = model.get_future_logits(past_hidden, future_offset=offset)  # [batch, 1, vocab]
            mtp_logits_list.append(future_logits.squeeze(1).float())  # [batch, vocab]
        
        next_token_logits = model.compute_contrast_logits(
            ntp_logits, 
            mtp_logits_list, 
            alpha=mtp_alpha,
            amateur_weights=amateur_weights
        )

        next_token_logits = next_token_logits.to(input_ids.device)
        next_token_scores = logits_processor(input_ids, next_token_logits)

        if return_dict_in_generate and output_scores:
            scores += (next_token_scores,)
        if return_dict_in_generate and output_logits:
            raw_logits += (ntp_logits,)

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
    if not hasattr(model, "get_future_logits"): 
        raise ValueError("Model does not support CMTP generation. Ensure it is an instance of LLMWithCMTP.")
    
    future_offsets_comb = [
        [[2], [1.0]],
        [[2, 3], [0.3, 0.7]],
        [[2, 3], [0.5, 0.5]],
        [[2, 3], [0.7, 0.3]],
        [[3], [1.0]],
        [[3, 4], [0.3, 0.7]],
        [[3, 4], [0.5, 0.5]],
        [[3, 4], [0.7, 0.3]],
        [[4], [1.0]],
        [[2, 3, 4], [0.5, 0.3, 0.2]],
        [[2, 3, 4], [0.3, 0.5, 0.2]],
        [[2, 3, 4], [0.2, 0.3, 0.5]],
        [[2, 3, 4], [0.34, 0.33, 0.33]]
    ]

    mtp_alpha = kwargs.pop("mtp_alpha", 0.5)
    future_offsets_comb_idx = kwargs.pop("future_offsets_comb_idx", 0) 
    
    assert 0 <= future_offsets_comb_idx < len(future_offsets_comb)
    future_offsets = future_offsets_comb[future_offsets_comb_idx][0]
    amateur_weights = future_offsets_comb[future_offsets_comb_idx][1]

    gen_config = kwargs.get("generation_config")
    
    if gen_config is None:
        gen_config = GenerationConfig.from_model_config(model.config)
        kwargs["generation_config"] = gen_config
    
    kwargs["generation_config"].mtp_alpha = mtp_alpha
    kwargs["generation_config"].future_offsets = future_offsets
    kwargs["generation_config"].amateur_weights = amateur_weights

    generation_outputs = GenerationMixin.generate(
        model, *args, custom_generate=_cmtp_decoding, **kwargs
    )
    return generation_outputs