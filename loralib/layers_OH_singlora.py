# @title loralib/layers_OH_singlora.py
# %%writefile /content/SingloRA_CLIP/loralib/layers_OH_singlora.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import lớp cha từ file gốc
from .layers_singlora import SingLoRALayer

class LinearOHsingLoRA(nn.Linear, SingLoRALayer):
    """
    Implements Orthogonal-Head SingLoRA (OH-SingLoRA).

    Đây là bước đầu tiên trong việc xây dựng DO-SingLoRA. Giai đoạn này tập trung
    vào việc triển khai loss điều chuẩn trực giao, trong khi vẫn giữ lại
    kiến trúc gating của GMH-SingLoRA.
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
        # Khởi tạo các lớp cha y hệt như LinearGMHSingLoRA
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
            # Đóng băng trọng số gốc
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


def calculate_ortho_loss(lora_A_heads: torch.Tensor) -> torch.Tensor:
    """
    Tính toán loss điều chuẩn trực giao cho các head của adapter.
    Loss này sẽ nhỏ khi các không gian con được sinh bởi các ma trận A_i là trực giao,
    và lớn khi chúng trùng lặp.

    Loss = sum_{i < j} || A_i^T @ A_j ||^2_F

    Args:
        lora_A_heads (torch.Tensor): Tensor chứa các ma trận A của các head.
                                     Shape: (num_heads, D, rank_per_head)

    Returns:
        torch.Tensor: Một scalar tensor chứa giá trị loss.
    """
    if lora_A_heads.shape[0] <= 1:
        return torch.tensor(0.0, device=lora_A_heads.device, dtype=lora_A_heads.dtype)

    num_heads = lora_A_heads.shape[0]

    ortho_loss = torch.tensor(0.0, device=lora_A_heads.device, dtype=lora_A_heads.dtype)

    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            A_i = lora_A_heads[i] # Shape: (D, rank)
            A_j = lora_A_heads[j] # Shape: (D, rank)
            product = A_i.T @ A_j
            loss_pair = torch.norm(product, p='fro')**2
            ortho_loss += loss_pair

    return ortho_loss