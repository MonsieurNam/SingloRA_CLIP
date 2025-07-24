# @title loralib/utils.py

import os
import torch
import torch.nn as nn
from typing import Dict, List

from .layers import LoRALayer, PlainMultiheadAttentionLoRA
from .layers_singlora import PlainMultiheadAttentionSingLoRA, LinearSingLoRA, LinearDySingLoRA, PlainMultiheadAttentionAdapter

# Các từ điển này được sao chép từ loralib/utils.py để giữ nguyên logic
INDEX_POSITIONS_TEXT = {
    'top1': [11], 'top2': [10, 11], 'top3': [9, 10, 11], 'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7], 'up': [8, 9, 10, 11], 'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5], 'all': list(range(12))
}
INDEX_POSITIONS_VISION = {
    'ViT-B/16': {'all': list(range(12)), 'bottom': [0,1,2,3], 'mid': [4,5,6,7], 'up': [8,9,10,11]},
    'ViT-B/32': {'all': list(range(12)), 'bottom': [0,1,2,3], 'mid': [4,5,6,7], 'up': [8,9,10,11]},
    'ViT-L/14': {'all': list(range(24)), 'half-bottom': list(range(12)), 'half-up': list(range(12, 24))}
}


INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    # Đóng băng tất cả các tham số trước
    for n, p in model.named_parameters():
        p.requires_grad = False
    
    # Mở khóa có chọn lọc cho các tham số adapter
    for n, p in model.named_parameters():
        if 'lora_' in n or 'scaler' in n:
            p.requires_grad = True
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def get_lora_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        # Chỉ lấy các tham số đã được bật requires_grad
        if param.requires_grad:
            params.append(param)
    return params


def apply_lora(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers


def save_lora(args, list_lora_layers):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        if 'q' in args.params:
            layer_weights['q_proj'] = {
                'w_lora_A': layer.q_proj.w_lora_A.data,
                'w_lora_B': layer.q_proj.w_lora_B.data
            }
        if 'k' in args.params:
            layer_weights['k_proj'] = {
                'w_lora_A': layer.k_proj.w_lora_A.data,
                'w_lora_B': layer.k_proj.w_lora_B.data
            }
        if 'v' in args.params:
            layer_weights['v_proj'] = {
                'w_lora_A': layer.v_proj.w_lora_A.data,
                'w_lora_B': layer.v_proj.w_lora_B.data
            }
        if 'o' in args.params:
            layer_weights['proj'] = {
                'w_lora_A': layer.proj.w_lora_A.data,
                'w_lora_B': layer.proj.w_lora_B.data
            }

        weights[f'layer_{i}'] = layer_weights

    metadata = {
        'r': args.r,
        'alpha': args.alpha,
        'encoder': args.encoder,
        'params': args.params,
        'position': args.position
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }

    # to manage names like ViT-B/16
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    save_dir = f'{args.save_path}/{backbone}/{args.dataset}/{args.shots}shots/seed{args.seed}'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{args.filename}.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')


def load_lora(args, list_lora_layers):
    # to manage names like ViT-B/16
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    load_path = f'{args.save_path}/{backbone}/{args.dataset}/{args.shots}shots/seed{args.seed}/{args.filename}.pt'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    if metadata['r'] != args.r:
        raise ValueError(
            f"r mismatch: expected {args.r}, found {metadata['r']}")
    if metadata['alpha'] != args.alpha:
        raise ValueError(
            f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
    if metadata['encoder'] != args.encoder:
        raise ValueError(
            f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
    if metadata['params'] != args.params:
        raise ValueError(
            f"Params mismatch: expected {args.params}, found {metadata['params']}")
    if metadata['position'] != args.position:
        raise ValueError(
            f"Position mismatch: expected {args.position}, found {metadata['position']}")

    weights = loaded_data['weights']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        if 'q' in args.params and 'q_proj' in layer_weights:
            layer.q_proj.w_lora_A.data.copy_(
                layer_weights['q_proj']['w_lora_A'])
            layer.q_proj.w_lora_B.data.copy_(
                layer_weights['q_proj']['w_lora_B'])
        if 'k' in args.params and 'k_proj' in layer_weights:
            layer.k_proj.w_lora_A.data.copy_(
                layer_weights['k_proj']['w_lora_A'])
            layer.k_proj.w_lora_B.data.copy_(
                layer_weights['k_proj']['w_lora_B'])
        if 'v' in args.params and 'v_proj' in layer_weights:
            layer.v_proj.w_lora_A.data.copy_(
                layer_weights['v_proj']['w_lora_A'])
            layer.v_proj.w_lora_B.data.copy_(
                layer_weights['v_proj']['w_lora_B'])
        if 'o' in args.params and 'proj' in layer_weights:
            layer.proj.w_lora_A.data.copy_(layer_weights['proj']['w_lora_A'])
            layer.proj.w_lora_B.data.copy_(layer_weights['proj']['w_lora_B'])

    print(f'LoRA weights loaded from {load_path}')

def apply_adapter(args, clip_model):
    """
    Hàm chung để áp dụng bất kỳ loại adapter nào (SingLoRA, DySingLoRA)
    vào mô hình CLIP.
    """
    # Map tên adapter từ args sang lớp tương ứng
    adapter_map = {
        'singlora': LinearSingLoRA,
        'dysinglora': LinearDySingLoRA
    }

    # Kiểm tra xem adapter có được hỗ trợ không
    if args.adapter not in adapter_map:
        raise ValueError(f"Adapter type '{args.adapter}' is not supported by this function.")

    linear_adapter_class = adapter_map[args.adapter]
    print(f">> Applying {args.adapter.upper()} adapter...")

    # Tạo một dict chứa các kwargs chung cho các lớp adapter
    adapter_kwargs = {
        'r': args.r,
        'lora_alpha': args.alpha,
        'ramp_up_steps': args.ramp_up_steps
    }

    # Lặp qua cả hai encoder
    for encoder_name in ['text', 'vision']:
        if args.encoder == encoder_name or args.encoder == 'both':
            print(f"   Processing {encoder_name.capitalize()} Encoder...")

            if encoder_name == 'text':
                encoder = clip_model.transformer
                indices = INDEX_POSITIONS_TEXT[args.position]
            else:
                encoder = clip_model.visual.transformer
                indices = INDEX_POSITIONS_VISION[args.backbone][args.position]

            for i, block in enumerate(encoder.resblocks):
                if i in indices:
                    # Thay thế MHA bằng lớp vỏ chung
                    new_mha = PlainMultiheadAttentionAdapter(
                        block.attn,
                        linear_adapter_class=linear_adapter_class,
                        **adapter_kwargs,
                        enable_lora=args.params
                    )
                    block.attn = new_mha

                    # Áp dụng cho các lớp MLP nếu được yêu cầu
                    if 'mlp' in args.params:
                        for mlp_layer_name, mlp_layer in block.mlp.named_children():
                            if isinstance(mlp_layer, nn.Linear):
                                new_linear_layer = linear_adapter_class(
                                    mlp_layer,
                                    **adapter_kwargs
                                )
                                setattr(block.mlp, mlp_layer_name, new_linear_layer)

    print("Finished applying adapters.")
    return [] # Không cần trả về list nữa

def save_adapter(args, model):
    """
    Hàm chung để lưu các trọng số có thể huấn luyện của adapter.
    """
    adapter_state_dict = {}
    
    # Thu thập các tham số cần lưu từ state_dict của mô hình
    for name, param in model.state_dict().items():
        # Dựa vào quy ước đặt tên để xác định tham số của adapter
        if 'lora_' in name or 'scaler' in name:
            adapter_state_dict[name] = param

    # Tạo metadata để xác minh khi tải
    metadata = {
        'adapter': args.adapter,
        'r': args.r,
        'alpha': args.alpha,
        'params': args.params,
        'position': args.position,
        'encoder': args.encoder,
        'backbone': args.backbone,
    }
    
    save_data = {
        'weights': adapter_state_dict,
        'metadata': metadata
    }
    
    # Tạo đường dẫn và lưu file
    backbone_str = args.backbone.replace('/', '')
    save_dir = os.path.join(args.save_path, args.adapter, backbone_str, args.dataset, f"{args.shots}shots", f"seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'adapter_weights.pt')
    
    torch.save(save_data, save_path)
    print(f"{args.adapter.upper()} weights saved to {save_path}")


def load_adapter(args, model):
    """
    Hàm chung để tải các trọng số adapter vào mô hình.
    """
    backbone_str = args.backbone.replace('/', '')
    load_path = os.path.join(args.save_path, args.adapter, backbone_str, args.dataset, f"{args.shots}shots", f"seed{args.seed}", 'adapter_weights.pt')
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Adapter weights not found at: {load_path}")
        
    loaded_data = torch.load(load_path, map_location='cpu')
    
    # Kiểm tra metadata (rất quan trọng để tránh lỗi)
    metadata = loaded_data.get('metadata', {})
    expected_metadata = {
        'adapter': args.adapter,
        'r': args.r,
        'alpha': args.alpha,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"Metadata mismatch for '{key}'! Expected '{value}', but found '{metadata.get(key)}' in checkpoint.")
    
    weights = loaded_data['weights']
    
    # Tải trọng số vào mô hình. `strict=False` là cần thiết vì chúng ta
    # chỉ tải một phần nhỏ của toàn bộ state_dict.
    incompatible_keys = model.load_state_dict(weights, strict=False)
    
    if incompatible_keys.missing_keys:
        print(f"Info: Some adapter keys were not found in the model state_dict (this is expected): {incompatible_keys.missing_keys[:5]}...")
    if incompatible_keys.unexpected_keys:
        print(f"Warning: The checkpoint contains unexpected keys not present in the model: {incompatible_keys.unexpected_keys}")
        
    print(f"{args.adapter.upper()} weights loaded from {load_path}")