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
        **kwargs # Thêm **kwargs để nhận các tham số thừa từ super()
    ):
        # Không còn super().__init__() ở đây
        self.r = r
        self.lora_alpha = lora_alpha
        self.ramp_up_steps = ramp_up_steps

        if self.r > 0:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # training_step sẽ được đăng ký trong lớp con (LinearSingLoRA)
        # vì lớp này không phải là nn.Module

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

            # ==================== THAY ĐỔI CUỐI CÙNG ====================
            # Khởi tạo A từ phân phối chuẩn với độ lệch chuẩn RẤT NHỎ
            # thay vì Kaiming. Điều này làm cho A @ A.T lúc đầu gần bằng 0.
            nn.init.normal_(self.lora_A, std=0.01)
            # ==========================================================

    def forward(self, x: torch.Tensor, **kwargs):
        # Chuyển dtype để tương thích
        dtype = x.dtype
        weight = self.weight.to(dtype)
        bias = self.bias.to(dtype) if self.bias is not None else None

        if self.r == 0:
            return F.linear(x, weight, bias)

        original_output = F.linear(x, weight, bias)

        if self.training:
            self.step()

        lora_A = self.lora_A.to(dtype)

        # ==================== THAY ĐỔI CUỐI CÙNG (CÔNG THỨC) =================
        # Loại bỏ hoàn toàn heuristic chuẩn hóa
        # Và giữ nguyên công thức gốc A @ A.T
        delta_W = lora_A @ lora_A.T
        delta_W_raw = self.lora_A @ self.lora_A.T
        if self.training and self.training_step % 50 == 0: # In mỗi 50 bước
          with torch.no_grad():
              w0_flat = self.weight.float().flatten()
              dw_flat = delta_W_raw.float().flatten()
              # Tính cosin tương đồng
              cosine_sim = F.cosine_similarity(w0_flat, dw_flat, dim=0).item()
              # print(f"Cosine Similarity (W₀, ΔW): {cosine_sim:.6f}")

        lora_adjustment = F.linear(x, delta_W) * self.u() * self.scaling
        # ===================================================================
        # if self.training and self.training_step % 50 == 0: # In mỗi 50 bước
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

        # Deconstruct and replace layers
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.out_proj.bias is not None)

        with torch.no_grad():
            # Tách trọng số từ in_proj_weight của MHA gốc
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

        # Thay thế bằng các lớp SingLoRA
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

        # Xử lý batch_first
        if self.batch_first:
            query, key, value = [x.transpose(0, 1) for x in (query, key, value)]

        # Lấy kích thước
        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]

        # 1. Chiếu Q, K, V -> Đây là nơi logic SingLoRA được áp dụng
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. Reshape cho multi-head attention
        q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        # Shape bây giờ là (bsz, num_heads, seq_len, head_dim)

        # 3. Tính điểm attention
        # (bsz, num_heads, tgt_len, head_dim) @ (bsz, num_heads, head_dim, src_len)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4. Áp dụng các mặt nạ
        if attn_mask is not None:
            # attn_mask có thể là (tgt_len, src_len) hoặc (bsz*num_heads, tgt_len, src_len)
            # Chúng ta cần đảm bảo nó có thể broadcast được
            if attn_mask.dim() == 2:
                attn_scores += attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                # Nếu là 3D, nó có thể đã được chuẩn bị sẵn
                attn_scores += attn_mask

        if key_padding_mask is not None:
            # key_padding_mask (bsz, src_len)
            # unsqueeze để nó broadcast được với attn_scores (bsz, num_heads, tgt_len, src_len)
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # 5. Softmax và Dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 6. Tính đầu ra
        attn_output = attn_weights @ v # (bsz, num_heads, tgt_len, head_dim)

        # 7. Reshape và chiếu đầu ra
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        # Xử lý batch_first cho đầu ra
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        # Trả về kết quả
        if need_weights:
            # Xử lý average_attn_weights
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                return attn_output, attn_weights.sum(dim=1) / self.num_heads
            else:
                return attn_output, attn_weights
        else:
            return attn_output, None
        
class LinearDySingLoRA(nn.Linear, SingLoRALayer):
    """
    Implements DyLoRA: A SingLoRA variant with dynamic, asymmetric scaling
    vectors (row_scaler, col_scaler) to break the symmetry constraint.
    """
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        ramp_up_steps: int = 1000,
        **kwargs # kwargs để tương thích
    ):
        # 1. Khởi tạo các thành phần cơ bản (giống hệt LinearSingLoRA)
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
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

            self.lora_A = nn.Parameter(
                torch.zeros(max(self.in_features, self.out_features), self.r)
            )
            nn.init.normal_(self.lora_A, std=0.01)

            self.row_scaler = nn.Parameter(torch.ones(self.out_features))
            self.col_scaler = nn.Parameter(torch.ones(self.in_features))

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
        row_scaler = self.row_scaler.to(dtype)
        col_scaler = self.col_scaler.to(dtype)

        # --- LOGIC CỦA DY-SING-LORA ---

        lora_A_out = lora_A[:self.out_features, :]
        lora_A_in = lora_A[:self.in_features, :]

        delta_W_symmetric = lora_A_out @ lora_A_in.T

        # Áp dụng scaling bất đối xứng để phá vỡ tính đối xứng
        # row_scaler.unsqueeze(1) có shape (out_features, 1)
        # col_scaler.unsqueeze(0) có shape (1, in_features)
        # Phép nhân element-wise (broadcasting) sẽ scale mỗi hàng và mỗi cột
        delta_W_asymmetric = delta_W_symmetric * row_scaler.unsqueeze(1) * col_scaler.unsqueeze(0)

        lora_adjustment = F.linear(x, delta_W_asymmetric) * self.u() * self.scaling

        return original_output + lora_adjustment

# ------------------------------------------------------------------------------
# Tái cấu trúc PlainMultiheadAttention... để linh hoạt hơn
# ------------------------------------------------------------------------------

class PlainMultiheadAttentionAdapter(nn.Module):
    """
    Một lớp vỏ MHA tổng quát có thể nhận BẤT KỲ lớp adapter tuyến tính nào.
    """
    def __init__(
            self,
            existing_mha: nn.MultiheadAttention,
            linear_adapter_class, # Tham số mới: lớp adapter sẽ được sử dụng
            r: int = 0,
            lora_alpha: int = 1,
            ramp_up_steps: int = 1000,
            enable_lora: list = ['q', 'k', 'v'],
            **kwargs
        ):
        super().__init__()

        self.embed_dim = existing_mha.embed_dim
        self.num_heads = existing_mha.num_heads
        # ... (code sao chép các thuộc tính khác không đổi) ...
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.batch_first = existing_mha.batch_first
        self.dropout = existing_mha.dropout

        # Tách và sao chép trọng số (không đổi)
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

        # Thay thế bằng các lớp adapter được truyền vào
        # Truyền tất cả các tham số cần thiết
        adapter_kwargs = {'r': r, 'lora_alpha': lora_alpha, 'ramp_up_steps': ramp_up_steps}

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

            # 1. Project Q, K, V -> This is where the SingLoRA logic is applied
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

            # 2. Reshape for multi-head attention
            q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            k = k.view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            v = v.view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

            # 3. Compute attention scores
            attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # 4. Apply masks
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_scores += attn_mask.unsqueeze(0).unsqueeze(0)
                else:
                    attn_scores += attn_mask

            if key_padding_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )

            # 5. Softmax and Dropout
            attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

            # 6. Compute final output
            attn_output = attn_weights @ v

            # 7. Reshape and project output
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