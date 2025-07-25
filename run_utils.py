# @title run_utils.py

import random
import argparse
import numpy as np
import torch

from lora import run_lora

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dtd')
    parser.add_argument('--shots', default=16, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    # Training arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    # --- START OF MODIFICATIONS FOR PHASE 2 ---

    # General Adapter arguments
    parser.add_argument('--adapter', type=str, default='lora', 
                        choices=['lora', 'singlora', 'dysinglora', 'mhsinglora', 'gmhsinglora'], # Thêm 'gmhsinglora'
                        help='The type of adapter to use for fine-tuning.')
    
    # ... (các đối số chung khác không đổi) ...
    
    # MH-SingLoRA-specific arguments
    parser.add_argument('--num_heads', type=int, default=2,
                        help='[MH-SingLoRA & G-MHSingLoRA only] Number of heads for the adapter.')

    # LoRA / SingLoRA arguments (shared)
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'],
                        help='Where to insert the adapter modules in the transformer blocks.')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both',
                        help='Which encoder to apply the adapter to.')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'],
                        help='List of attention matrices to adapt (q, k, v, o).')
    parser.add_argument('--r', default=2, type=int,
                        help='The rank of the low-rank matrices.')
    parser.add_argument('--alpha', default=1, type=int,
                        help='The scaling factor (lora_alpha).')

    # LoRA-specific arguments
    parser.add_argument('--dropout_rate', default=0.25, type=float,
                        help='[LoRA only] Dropout rate applied before the LoRA module.')

    # SingLoRA-specific arguments
    parser.add_argument('--ramp_up_steps', type=int, default=1000,
                        help='[SingLoRA only] Number of ramp-up steps (T) for the adapter.')

    # --- END OF MODIFICATIONS FOR PHASE 2 ---

    parser.add_argument('--save_path', default=None, help='Path to save the adapter modules after training. Not saved if None.')
    parser.add_argument('--filename', default='adapter_weights', help='File name to save the adapter weights (.pt extension will be added).')

    parser.add_argument('--eval_only', default=False, action='store_true', help='Only evaluate the adapter modules (save_path should not be None).')
    args = parser.parse_args()

    return args


