
import json
import torch
import torch.distributed as dist
import os
from transformers.configuration_utils import PretrainedConfig
from typing import List, Union, Optional, Tuple, Mapping, Dict

def print_gpu_memory(device=0):
    if torch.cuda.is_available():
        print(f"--- GPU Memory (device {device}) ---")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"Reserved:  {torch.cuda.memory_reserved(device)  / 1024**2:.2f} MB")
        print(f"Max Alloc: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
        print("-------------------------------")
    else:
        print("CUDA not available")
        
class LatentAttentionConfig(PretrainedConfig):
    model_type = 'latent_attention'
    is_composition = False
    _name_or_path = "latent_attention"

    def __init__(
        self,
        num_latents_value: int=64,
        num_cross_heads: int=12,
        output_normalize: bool=True,
        hidden_dim: int=768,
        latent_dim: int=768,
        cross_dim_head: int=768,
        **kwargs,
    ):
        self.num_latents_value = num_latents_value
        self.num_cross_heads = num_cross_heads
        self.output_normalize = output_normalize
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cross_dim_head = cross_dim_head

        super().__init__(**kwargs)
        


def full_contrastive_scores_and_labels(
        query: torch.Tensor,
        key: torch.Tensor,
        use_all_pairs: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    assert key.shape[0] % query.shape[0] == 0, '{} % {} > 0'.format(key.shape[0], query.shape[0])

    train_n_passages = key.shape[0] // query.shape[0]
    
    labels = torch.arange(0, query.shape[0], dtype=torch.long, device=query.device)
    labels = labels * train_n_passages

    # batch_size x (batch_size x n_psg)
    qk = torch.mm(query, key.t())

    if not use_all_pairs:
        return qk, labels

    # batch_size x dim
    sliced_key = key.index_select(dim=0, index=labels)
    assert query.shape[0] == sliced_key.shape[0]

    # batch_size x batch_size
    kq = torch.mm(sliced_key, query.t())
    kq.fill_diagonal_(float('-inf'))

    qq = torch.mm(query, query.t())
    qq.fill_diagonal_(float('-inf'))

    kk = torch.mm(sliced_key, sliced_key.t())
    kk.fill_diagonal_(float('-inf'))

    scores = torch.cat([qk, kq, qq, kk], dim=-1)

    return scores, labels