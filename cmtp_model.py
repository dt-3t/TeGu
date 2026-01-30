import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.generation.utils import GenerationMixin

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

@dataclass
class MTPModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mtp_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    offset_losses: Optional[dict] = None

class MTPModule(nn.Module):
    def __init__(self, config, max_future_step=16, intermediate_factor=2.7):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_size
        inter_dim = int(hidden_dim * intermediate_factor)
        
        self.step_embedding = nn.Embedding(max_future_step + 1, hidden_dim)
        
        self.ada_lin = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True) 
        )

        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.norm = nn.RMSNorm(hidden_dim, eps=eps) 
        
        self.gate_proj = nn.Linear(hidden_dim, inter_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, hidden_dim, bias=False)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.step_embedding.weight, std=0.02)
        
        for m in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        
        nn.init.zeros_(self.ada_lin[-1].weight)
        nn.init.zeros_(self.ada_lin[-1].bias)
        
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, hidden_states, step_ids):
        step_emb = self.step_embedding(step_ids) 
        style = self.ada_lin(step_emb)
        
        gamma, beta = style.chunk(2, dim=-1)
        gamma = gamma.view(1, 1, -1)
        beta = beta.view(1, 1, -1)
        
        normed_x = self.norm(hidden_states)
        x = normed_x * (1 + gamma) + beta
        
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        return x


class LLMWithCMTP(nn.Module):
    def __init__(self, model_path, future_offsets=[2, 3, 4], kd_alpha=0.5, kd_temperature=2.0, **kwargs):
        # In the paper, 1 denotes the next-next token, whereas in the code, 2 represents the next-next token. Please note this distinction.
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.config = self.base_model.config
        
        self.future_offsets = future_offsets
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature

        print(f"{GREEN}Init LLMWithCMTP with future_offsets={future_offsets}{RESET}")
        
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        self.max_future_step = 16
        self.mtp_module = MTPModule(self.config, max_future_step=self.max_future_step)
        
        self.mtp_module.to(dtype=self.base_model.dtype)

    @property
    def device(self):
        return self.base_model.device

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def generate(self, *args, **kwargs):
        return GenerationMixin.generate(self, *args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    def _compute_single_offset_loss(self, curr_hidden, curr_labels, curr_teacher_logits, step_idx):
        mtp_features = self.mtp_module(curr_hidden, step_idx)
        student_logits = self.base_model.lm_head(mtp_features)
        
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = student_logits.view(-1, self.config.vocab_size)
        shift_labels = curr_labels.view(-1)
        hard_loss = loss_fct(shift_logits, shift_labels)

        kd_loss = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
        if self.kd_alpha > 0:
            t_flat = curr_teacher_logits.reshape(-1, curr_teacher_logits.size(-1))
            s_flat = student_logits.reshape(-1, student_logits.size(-1))
            
            num_tokens = t_flat.size(0)
            chunk_size = 2048 
            
            chunk_kl_sum = 0.0
            for i in range(0, num_tokens, chunk_size):
                end_i = min(i + chunk_size, num_tokens)
                c_t = t_flat[i:end_i]
                c_s = s_flat[i:end_i]
                
                with torch.no_grad():
                    c_t_probs = F.softmax(c_t / self.kd_temperature, dim=-1)
                
                c_s_log_probs = F.log_softmax(c_s / self.kd_temperature, dim=-1)
                
                chunk_kl_sum += F.kl_div(
                    c_s_log_probs, 
                    c_t_probs, 
                    reduction="sum"
                ) * (self.kd_temperature ** 2)
            
            kd_loss = chunk_kl_sum / num_tokens

        step_loss = (1 - self.kd_alpha) * hard_loss + self.kd_alpha * kd_loss
        
        return step_loss, hard_loss.detach(), kd_loss.detach()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=use_cache,
                labels=labels,
                return_dict=True,
                **kwargs
            )
        
        teacher_logits = outputs.logits
        last_hidden_state = outputs.hidden_states[-1]
        
        total_mtp_loss = 0.0
        n_offsets_computed = 0
        loss_monitor_dict = {}
        
        if labels is not None:
            target_device = last_hidden_state.device
            if self.mtp_module.gate_proj.weight.device != target_device:
                self.mtp_module.to(target_device)

            for offset in self.future_offsets:
                valid_len = last_hidden_state.shape[1] - offset
                if valid_len <= 0:
                    continue

                curr_hidden = last_hidden_state[:, :valid_len, :].contiguous().detach()
                curr_hidden.requires_grad = True
                
                curr_labels = labels[:, offset:].contiguous()
                curr_teacher_logits = teacher_logits[:, offset - 1 : -1, :].contiguous()

                step_idx = torch.tensor([offset], device=target_device, dtype=torch.long)
                
                step_loss, step_hard, step_kd = checkpoint.checkpoint(
                    self._compute_single_offset_loss,
                    curr_hidden,         
                    curr_labels,         
                    curr_teacher_logits, 
                    step_idx,            
                    use_reentrant=False
                )

                total_mtp_loss += step_loss
                n_offsets_computed += 1

                #loss_monitor_dict[f"loss_off{offset}"] = step_loss.detach()
                loss_monitor_dict[f"hard_off{offset}"] = step_hard.detach()
                loss_monitor_dict[f"kd_off{offset}"] = step_kd.detach()
        
        final_mtp_loss = total_mtp_loss / max(1, n_offsets_computed) if labels is not None else None

        return MTPModelOutput(
            loss=final_mtp_loss,
            logits=teacher_logits,
            mtp_logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            offset_losses=loss_monitor_dict
        )
    
    def get_future_logits(self, hidden_states, future_offset):
        """
        Called during inference.
        Input:
            hidden_states: [batch_size, len, hidden_dim]
            future_offset: the number of future steps to predict (int). Since 1 means next-token (handled by NTP), this supports from 2, where 2 means predicting the next-next-token.
        Output:
            logits [batch, len, vocab]
        """
        assert future_offset >= 2, "future_offset should be >= 2"
        target_device = hidden_states.device
        
        if self.mtp_module.gate_proj.weight.device != target_device:
            self.mtp_module.to(target_device)

        step_tensor = torch.tensor([future_offset], device=target_device, dtype=torch.long)
        mtp_features = self.mtp_module(hidden_states, step_tensor)
        logits = self.base_model.lm_head(mtp_features)
             
        return logits
    
    def compute_contrast_logits(self, ntp_logits, mtp_logits_list, alpha=None, beta=None, amateur_weights=None):
        curr_alpha = alpha if alpha is not None else getattr(self, "alpha", 0.5)
        curr_beta = beta if beta is not None else getattr(self, "beta", 0.0)

        expert_logits = ntp_logits
        amateur_logits = mtp_logits_list

        expert_log_probs = F.log_softmax(expert_logits, dim=-1)  # [batch_size, vocab_size]

        if len(amateur_logits) > 1:
            amateur_log_probs_stack = torch.stack(
                [F.log_softmax(logits, dim=-1) for logits in amateur_logits], 
                dim=0
            )  # [num_amateurs, batch_size, vocab_size]
            
            target_device = expert_logits.device
            num_amateurs = len(amateur_logits)

            if len(amateur_weights) != num_amateurs:
                raise ValueError(f"Weights length {len(amateur_weights)} != logits length {num_amateurs}")
            
            w_tensor = torch.tensor(amateur_weights, device=target_device, dtype=expert_logits.dtype)
            w_normalized = w_tensor / (w_tensor.sum() + 1e-8)
            log_w = torch.log(w_normalized + 1e-10).view(-1, 1, 1) 
            
            amateur_log_probs = torch.logsumexp(amateur_log_probs_stack + log_w, dim=0)
            
        else:
            amateur_log_probs = F.log_softmax(amateur_logits[0], dim=-1)

        # CD formula
        diff_log_probs = (1 + curr_alpha) * expert_log_probs - curr_alpha * amateur_log_probs

        if curr_beta > 0:
            expert_probs = torch.exp(expert_log_probs)
            expert_max_probs, _ = torch.max(expert_probs, dim=-1, keepdim=True)
            mask = expert_probs < (curr_beta * expert_max_probs)
            diff_log_probs = diff_log_probs.masked_fill(mask, -float('inf'))

        return diff_log_probs
    
    @property
    def output_keys(self):
        return ["loss", "logits", "mtp_logits", "past_key_values", "hidden_states", "attentions"]
    
    def state_dict(self, *args, **kwargs):
        return self.mtp_module.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        return self.mtp_module.load_state_dict(state_dict, strict=strict)
    
    def load_mtp_projector(self, ckpt_path, map_location="cpu", strict=True):
        state_dict = torch.load(ckpt_path, map_location=map_location)
        missing, unexpected = self.mtp_module.load_state_dict(
            state_dict, strict=strict
        )
        print(f"{GREEN}Loaded MTP module from {ckpt_path}{RESET}")
        if missing:
            print(f"{YELLOW}Missing keys: {missing}{RESET}")
        if unexpected:
            print(f"{YELLOW}Unexpected keys: {unexpected}{RESET}")