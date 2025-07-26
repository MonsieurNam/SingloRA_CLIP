# @title loralib/layers_singlora.py
#  [SingLoRA] - layers_singlora.py
#  This code implements the SingLoRA adapter layers based on the paper:
#  "SingLoRA: Low Rank Adaptation Using a Single Matrix"
#
#  Implementation is based on the existing LoRA structure for easy integration.
#  Author: [Your Name/Alias]
#  Date: [Current Date]
#  ------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple


class SingLoRALayer:
    """
    A mixin class for SingLoRA layers. This class handles the core logic of SingLoRA,
    including the ramp-up function u(t), scaling, and managing the training step.
    It does NOT inherit from nn.Module.
    """
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        ramp_up_steps: int,
        **kwargs 
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.ramp_up_steps = ramp_up_steps

        if self.r > 0:
            self.scaling = self.lora_alpha / math.sqrt(self.r)


    def u(self) -> float:
        if self.ramp_up_steps == 0:
            return 1.0
        # self.training_step sẽ được truy cập từ đối tượng lớp con
        return min(1.0, self.training_step.item() / self.ramp_up_steps)

    def step(self, steps: int = 1):
        self.training_step += steps


class LinearSingLoRA(nn.Linear, SingLoRALayer):
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        ramp_up_steps: int = 1000,
        **kwargs
    ):
        nn.Linear.__init__(
            self,
            in_features=existing_linear.in_features,
            out_features=existing_linear.out_features,
            bias=existing_linear.bias is not None
        )
        SingLoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, ramp_up_steps=ramp_up_steps)
        self.register_buffer('training_step', torch.tensor(0, dtype=torch.long))
        self.load_state_dict(existing_linear.state_dict(), strict=False)

        if self.r > 0:
            assert self.in_features == self.out_features, \
                "This implementation currently only supports square matrices."
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

            self.lora_A = nn.Parameter(torch.zeros(self.out_features, self.r))

            nn.init.normal_(self.lora_A, std=0.01)

    def forward(self, x: torch.Tensor, **kwargs):
        dtype = x.dtype
        weight = self.weight.to(dtype)
        bias = self.bias.to(dtype) if self.bias is not None else None

        if self.r == 0:
            return F.linear(x, weight, bias)

        original_output = F.linear(x, weight, bias)

        if self.training:
            self.step()

        lora_A = self.lora_A.to(dtype)

        delta_W = lora_A @ lora_A.T
        delta_W_raw = self.lora_A @ self.lora_A.T
        if self.training and self.training_step % 50 == 0:
          with torch.no_grad():
              w0_flat = self.weight.float().flatten()
              dw_flat = delta_W_raw.float().flatten()
              cosine_sim = F.cosine_similarity(w0_flat, dw_flat, dim=0).item()
              # print(f"Cosine Similarity (W₀, ΔW): {cosine_sim:.6f}")

        lora_adjustment = F.linear(x, delta_W) * self.u() * self.scaling
        # ===================================================================
        # if self.training and self.training_step % 50 == 0: 
        #     print(f"\n--- LOG (Step {self.training_step.item()}) ---")
        #     print(f"Norm of lora_A: {torch.linalg.norm(self.lora_A.float()).item():.6f}")
        #     print(f"---------------------------\n")

        return original_output + lora_adjustment


class PlainMultiheadAttentionSingLoRA(nn.Module):
    """
    Final, robust implementation of the SingLoRA MHA wrapper.
    This version directly implements the attention mechanism to ensure shape correctness.
    """
    def __init__(
            self,
            existing_mha: nn.MultiheadAttention,
            r: int = 0,
            lora_alpha: int = 1,
            ramp_up_steps: int = 1000,
            enable_lora: list = ['q', 'k', 'v'],
            **kwargs
        ):
        super().__init__()

        self.embed_dim = existing_mha.embed_dim
        self.num_heads = existing_mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.batch_first = existing_mha.batch_first
        self.dropout = existing_mha.dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.out_proj.bias is not None)

        with torch.no_grad():
            q_w, k_w, v_w = existing_mha.in_proj_weight.chunk(3)
            self.q_proj.weight.copy_(q_w)
            self.k_proj.weight.copy_(k_w)
            self.v_proj.weight.copy_(v_w)

            if existing_mha.in_proj_bias is not None:
                q_b, k_b, v_b = existing_mha.in_proj_bias.chunk(3)
                self.q_proj.bias.copy_(q_b)
                self.k_proj.bias.copy_(k_b)
                self.v_proj.bias.copy_(v_b)

            self.out_proj.load_state_dict(existing_mha.out_proj.state_dict())

        if 'q' in enable_lora:
            self.q_proj = LinearSingLoRA(self.q_proj, r, lora_alpha, ramp_up_steps)
        if 'k' in enable_lora:
            self.k_proj = LinearSingLoRA(self.k_proj, r, lora_alpha, ramp_up_steps)
        if 'v' in enable_lora:
            self.v_proj = LinearSingLoRA(self.v_proj, r, lora_alpha, ramp_up_steps)
        if 'o' in enable_lora:
            self.out_proj = LinearSingLoRA(self.out_proj, r, lora_alpha, ramp_up_steps)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if self.batch_first:
            query, key, value = [x.transpose(0, 1) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_scores += attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                attn_scores += attn_mask

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = attn_weights @ v 

        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                return attn_output, attn_weights.sum(dim=1) / self.num_heads
            else:
                return attn_output, attn_weights
        else:
            return attn_output, None
       
class LinearGMHSingLoRA(nn.Linear, SingLoRALayer):
    """
    Implements Gated Multi-Head SingLoRA (G-MHSingLoRA).

    Each head's contribution is dynamically weighted by a gating network
    that takes the input 'x' as context. This allows the adapter to
    selectively use different "expert" heads for different inputs.
    """
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 2,
        lora_alpha: int = 1,
        ramp_up_steps: int = 100,
        num_heads: int = 2,
        **kwargs
    ):
        nn.Linear.__init__(
            self,
            in_features=existing_linear.in_features,
            out_features=existing_linear.out_features,
            bias=existing_linear.bias is not None
        )
        SingLoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, ramp_up_steps=ramp_up_steps)
        
        self.num_heads = num_heads
        if self.r > 0:
            assert r % num_heads == 0, f"Rank 'r' ({r}) must be divisible by 'num_heads' ({num_heads})."
            self.rank_per_head = r // num_heads
        
        self.register_buffer('training_step', torch.tensor(0, dtype=torch.long))
        self.load_state_dict(existing_linear.state_dict(), strict=False)

        if self.r > 0:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
            
            self.lora_A_heads = nn.Parameter(
                torch.zeros(
                    self.num_heads,
                    max(self.in_features, self.out_features),
                    self.rank_per_head
                )
            )
            nn.init.normal_(self.lora_A_heads, std=0.01)
            
            self.gating_network = nn.Sequential(
                nn.Linear(self.in_features, self.num_heads, bias=False),
                nn.Softmax(dim=-1)
            )
            nn.init.zeros_(self.gating_network[0].weight)
            
    def forward(self, x: torch.Tensor, **kwargs):
        dtype = x.dtype
        weight = self.weight.to(dtype)
        bias = self.bias.to(dtype) if self.bias is not None else None
        
        if self.r == 0:
            return F.linear(x, weight, bias)

        original_output = F.linear(x, weight, bias)
        
        if self.training:
            self.step()
        
        gating_weights = self.gating_network(x.detach().to(self.gating_network[0].weight.dtype))
        gating_weights = gating_weights.to(dtype)

        lora_A_heads = self.lora_A_heads.to(dtype)
        lora_A_heads_out = lora_A_heads[:, :self.out_features, :]
        lora_A_heads_in = lora_A_heads[:, :self.in_features, :]
        all_delta_W = lora_A_heads_out @ lora_A_heads_in.transpose(1, 2) 
        lora_adjustment = torch.einsum('...i,hio,...h->...o', x, all_delta_W, gating_weights)
        final_adjustment = lora_adjustment * self.u() * self.scaling
            
        return original_output + final_adjustment

class PlainMultiheadAttentionAdapter(nn.Module):
    """
    Một lớp vỏ MHA tổng quát có thể nhận BẤT KỲ lớp adapter tuyến tính nào.
    """
    def __init__(
            self,
            existing_mha: nn.MultiheadAttention,
            linear_adapter_class, 
            r: int = 0,
            lora_alpha: int = 1,
            ramp_up_steps: int = 1000,
            enable_lora: list = ['q', 'k', 'v'],
            **kwargs
        ):
        super().__init__()

        self.embed_dim = existing_mha.embed_dim
        self.num_heads = existing_mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.batch_first = existing_mha.batch_first
        self.dropout = existing_mha.dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.out_proj.bias is not None)
        with torch.no_grad():
            q_w, k_w, v_w = existing_mha.in_proj_weight.chunk(3)
            self.q_proj.weight.copy_(q_w)
            self.k_proj.weight.copy_(k_w)
            self.v_proj.weight.copy_(v_w)
            if existing_mha.in_proj_bias is not None:
                q_b, k_b, v_b = existing_mha.in_proj_bias.chunk(3)
                self.q_proj.bias.copy_(q_b)
                self.k_proj.bias.copy_(k_b)
                self.v_proj.bias.copy_(v_b)
            self.out_proj.load_state_dict(existing_mha.out_proj.state_dict())

        adapter_kwargs = {
            'r': r,
            'lora_alpha': lora_alpha,
            'ramp_up_steps': ramp_up_steps
        }
        
        adapter_kwargs.update(kwargs)

        if 'q' in enable_lora:
            self.q_proj = linear_adapter_class(self.q_proj, **adapter_kwargs)
        if 'k' in enable_lora:
            self.k_proj = linear_adapter_class(self.k_proj, **adapter_kwargs)
        if 'v' in enable_lora:
            self.v_proj = linear_adapter_class(self.v_proj, **adapter_kwargs)
        if 'o' in enable_lora:
            self.out_proj = linear_adapter_class(self.out_proj, **adapter_kwargs)
    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

            if self.batch_first:
                query, key, value = [x.transpose(0, 1) for x in (query, key, value)]

            tgt_len, bsz, embed_dim = query.shape
            src_len = key.shape[0]

            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

            q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            k = k.view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            v = v.view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

            attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_scores += attn_mask.unsqueeze(0).unsqueeze(0)
                else:
                    attn_scores += attn_mask

            if key_padding_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )

            attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

            attn_output = attn_weights @ v

            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)
            attn_output = self.out_proj(attn_output)

            if self.batch_first:
                attn_output = attn_output.transpose(0, 1)

            if need_weights:
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                if average_attn_weights:
                    return attn_output, attn_weights.sum(dim=1) / self.num_heads
                else:
                    return attn_output, attn_weights
            else:
                return attn_output, None