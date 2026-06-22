import json
import torch
import torch.distributed as dist
from config import Arguments
import os

from typing import List, Union, Optional, Tuple, Mapping, Dict, Sequence


def save_json_to_file(objects: Union[List, dict], path: str, line_by_line: bool = False):
    if line_by_line:
        assert isinstance(objects, list), 'Only list can be saved in line by line format'

    with open(path, 'w', encoding='utf-8') as writer:
        if not line_by_line:
            json.dump(objects, writer, ensure_ascii=False, indent=4, separators=(',', ':'))
        else:
            for obj in objects:
                writer.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')))
                writer.write('\n')


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def dist_gather_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None

    t = t.contiguous()
    all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(all_tensors, t)

    all_tensors[dist.get_rank()] = t
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors

def collect_blocks_concat_rows(global_tensor: torch.Tensor, B: int, P: int, G: int,
                               gpu_ids: Optional[Sequence[int]] = None, full_contrastive=False) -> torch.Tensor:
    """
    Collects each GPU's (B, B*P) block from the global tensor and concatenates them
    along the x-axis (rows).

    Args:
        global_tensor: Tensor of shape (G*B, G*B*P).
        B: batch_size.
        P: num_passages.
        G: num_gpus (total GPUs assumed in the global tensor layout).
        gpu_ids: optional list/sequence of gpu ids to collect (default: all 0..G-1).

    Returns:
        concatenated tensor of shape (len(gpu_ids_or_all)*B, B*P).
        If gpu_ids is None this will be (G*B, B*P).
    """
    B = int(B)
    P = int(P)
    G = int(G)
    expected_shape = (G * B, G * B * P)
    if full_contrastive==True:
        assert global_tensor.shape == expected_shape, f"unexpected global shape {global_tensor.shape}, expected {expected_shape}"

    if gpu_ids is None:
        gpu_ids = range(G)

    blocks = []
    for gid in gpu_ids:
        if not (0 <= gid < G):
            raise ValueError(f"gpu id {gid} out of range [0, {G-1}]")
        x0 = gid * B
        x1 = x0 + B
        y0 = gid * (B * P)
        y1 = y0 + B * P
        blocks.append(global_tensor[x0:x1, y0:y1])

    # concatenate along rows (x axis)
    return torch.cat(blocks, dim=0)

@torch.no_grad()
def select_grouped_indices(scores: torch.Tensor,
                           group_size: int,
                           start: int = 0) -> torch.Tensor:
    assert len(scores.shape) == 2
    batch_size = scores.shape[0]
    assert batch_size * group_size <= scores.shape[1]

    indices = torch.arange(0, group_size, dtype=torch.long)
    indices = indices.repeat(batch_size, 1)
    indices += torch.arange(0, batch_size, dtype=torch.long).unsqueeze(-1) * group_size
    indices += start

    return indices.to(scores.device)


def full_contrastive_scores_and_labels(
        query: torch.Tensor,
        key: torch.Tensor,
        use_all_pairs: bool = False,
        fill_inf=True) -> Tuple[torch.Tensor, torch.Tensor]:
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
    qq = torch.mm(query, query.t())
    kk = torch.mm(sliced_key, sliced_key.t())

    if fill_inf:
        # in distillation, it is not necessary to fill inf values
        kq.fill_diagonal_(float('-inf'))
        qq.fill_diagonal_(float('-inf'))
        kk.fill_diagonal_(float('-inf'))

    scores = torch.cat([qk, kq, qq, kk], dim=-1)

    return scores, labels

def full_contrastive_scores_and_labels_kd(
        query: torch.Tensor,
        key: torch.Tensor,
        use_all_pairs: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
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
    qq = torch.mm(query, query.t())
    kk = torch.mm(sliced_key, sliced_key.t())

    kq = remove_diagonal(kq.size(0), kq)
    qq = remove_diagonal(qq.size(0), qq)
    kk = remove_diagonal(kk.size(0), kk)

    scores = torch.cat([qk, kq, qq, kk], dim=-1)

    return scores, labels

def remove_diagonal(n, tensor):
    return tensor.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)

def slice_batch_dict(batch_dict: Dict[str, torch.Tensor], prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in batch_dict.items() if k.startswith(prefix)}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, round_digits: int = 3):
        self.name = name
        self.round_digits = round_digits
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return '{}: {}'.format(self.name, round(self.avg, self.round_digits))


if __name__ == '__main__':
    query = torch.randn(4, 16)
    key = torch.randn(4 * 3, 16)
    scores, labels = full_contrastive_scores_and_labels(query, key)
    print(scores.shape)
    print(labels)


def save_sentence_encoder_config(model, output_dir, args : Arguments):
    if not args.should_save:
        return
    max_seq_length = max( args.q_max_len, args.p_max_len)
    pooling_mode_cls_token = str(args.pooling_source=='cls').lower()
    pooling_mode_mean_tokens = str(args.pooling_source=='mean').lower()
    if args.teacher:
        word_embedding_dimension = model.lm_p.config.hidden_size
    else:
        word_embedding_dimension = model.model.base_model.config.hidden_size

    modules='''[
  {
    "idx": 0,
    "name": "0",
    "path": "",
    "type": "sentence_transformers.models.Transformer"
  },
  {
    "idx": 1,
    "name": "1",
    "path": "1_Pooling",
    "type": "sentence_transformers.models.Pooling"
  },
  {
    "idx": 2,
    "name": "2",
    "path": "2_Normalize",
    "type": "sentence_transformers.models.Normalize"
  }
]'''

    bert=f'''{{
  "max_seq_length": 512,
  "do_lower_case": false
}}'''

    pool=f'''{{
  "word_embedding_dimension": {word_embedding_dimension},
  "pooling_mode_cls_token": {pooling_mode_cls_token},
  "pooling_mode_mean_tokens": {pooling_mode_mean_tokens},
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false
}}'''

    with open(output_dir + '/modules.json', 'wt') as modulesOut:
        print(modules, file=modulesOut)
    with open(output_dir+'/sentence_bert_config.json', 'wt') as bertOut:
        print(bert, file=bertOut)
    os.makedirs(output_dir+'/1_Pooling')
    with open(output_dir+'/1_Pooling/config.json', 'wt') as poolOut:
        print(pool, file=poolOut)

def last_token_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    # left_padding = True
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def mean_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    # left_padding = True
    assert left_padding, "support only left padding for now"
    seq_lengths = attention_mask.sum(dim=1) #mean pooling includes the EOS token
    return torch.stack(
        [
            last_hidden_states[i, -length:, :].mean(dim=0)
            for i, length in enumerate(seq_lengths)
        ],
        dim=0,
    )