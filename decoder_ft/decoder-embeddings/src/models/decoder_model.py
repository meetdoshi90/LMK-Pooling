import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from torch import Tensor
from collections import defaultdict
from transformers import (
    AutoModel,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    Mxfp4Config,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import ModelOutput
import gc
from config import Arguments
from logger_config import logger
from utils import dist_gather_tensor, select_grouped_indices, full_contrastive_scores_and_labels, last_token_pool, full_contrastive_scores_and_labels_kd, mean_pool, collect_blocks_concat_rows
import torch.distributed as dist
from transformers.masking_utils import create_causal_mask
from transformers.models.mistral.modeling_mistral import (
    MistralPreTrainedModel,
    MistralDecoderLayer,
    MistralRMSNorm,
    MistralRotaryEmbedding,
    MistralConfig,
    DynamicCache,
    BaseModelOutputWithPast
)


@dataclass
class DecoderOutput(ModelOutput):
    loss: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    scores: Optional[Tensor] = None


def print_gpu_state():
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev)
        reserved = torch.cuda.memory_reserved(dev)
        free, total = torch.cuda.mem_get_info()
        print(f"Allocated: {alloc/1024**2:.2f} MB")
        print(f"Reserved:  {reserved/1024**2:.2f} MB")
        print(f"Free:      {free/1024**2:.2f} MB")
        print(f"Total VRAM:{total/1024**2:.2f} MB")


# ── NEW: parse the comma-separated granularity set once ───────────────────────
def _parse_granularity_set(raw: Optional[str]) -> Optional[List[int]]:
    """'32,64,128' -> [32, 64, 128]. Returns None if raw is None/empty."""
    if not raw:
        return None
    return [int(g.strip()) for g in raw.split(",") if g.strip()]


class DecoderModel(nn.Module):
    def __init__(self, args: Arguments):
        super().__init__()
        if 'gpt-oss' in args.model_name_or_path:
            quantization_config = Mxfp4Config(dequantize=True)
            model_kwargs = dict(
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
                use_cache=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

        if args.use_reranker_for_decoder_distillation is not None:
            assert args.do_kd_biencoder == False, \
                'why do you want to kd scores when you are already using a reranker ;)'
            logger.info(f'Using reranker for distillation {args.use_reranker_for_decoder_distillation}')
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                args.use_reranker_for_decoder_distillation)
            self.reranker_model.eval()
        else:
            self.reranker_model = None

        print("#" * 100)
        print('model loaded')
        print("#" * 100)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.args = args
        self.pooler = nn.Linear(self.model.config.hidden_size, args.out_dimension) if args.add_pooler else nn.Identity()
        self.multi_layer_loss = args.multi_layer_loss
        self.multi_layer_loss_layers = args.multi_layer_loss_layers
        self.multi_layer_loss_scale = args.multi_layer_loss_scale
        self.self_distil_recursive = args.self_distil_recursive
        self.self_distil = args.self_distil
        self.self_distil_src_layer = args.self_distil_src_layer
        self.self_distil_tgt_layer = args.self_distil_tgt_layer
        self.num_layers = self.model.config.num_hidden_layers
        self.cache_minibatch_size = self.args.cache_minibatch_size
        self.document_cache_minibatch_size = self.args.document_cache_minibatch_size

        # ── NEW: LMK pooling setup ─────────────────────────────────────────────
        self._lmk_granularity_set: Optional[List[int]] = _parse_granularity_set(
            getattr(args, 'lmk_granularity_set', None)
        )
        self._lmk_granularity: int = getattr(args, 'lmk_granularity', 64)
        # Use EOS as the landmark token — already in vocab, no resize needed
        if getattr(args, 'pooling_source', '') == 'lmk':
            self._lmk_token_id: int = self.model.config.eos_token_id
            # Also write back to args so the data loader picks it up
            args.lmk_token_id = self._lmk_token_id
            logger.info(f"LMK pooling: using eos_token_id={self._lmk_token_id} as landmark token")
        else:
            self._lmk_token_id: int = getattr(args, 'lmk_token_id', -1)

        print("multilayer args", self.multi_layer_loss, self.multi_layer_loss_layers,
              self.multi_layer_loss_scale, self.num_layers)

        from trainers import BiencoderTrainer
        self.trainer: Optional[BiencoderTrainer] = None

    # ── NEW: core LMK pooling helper ───────────────────────────────────────────
    def _lmk_pool(
        self,
        hidden_states: Tensor,   # (B, T, D)
        input_ids: Tensor,       # (B, T)
        attention_mask: Tensor,  # (B, T)
    ) -> Tensor:                 # (B, D)
        """
        Mean-pool over positions where the token is [LMK] and not padding.

        For sequences that contain no [LMK] token (e.g. very short sequences
        where the first LMK was trimmed), falls back to standard attention-mask
        mean pool so training never crashes.
        """
        if self._lmk_token_id < 0:
            raise RuntimeError("_lmk_token_id not set — check pooling_source config")

        lmk_mask = (input_ids == self._lmk_token_id) & attention_mask.bool()  # (B, T)
        lmk_mask_f = lmk_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)     # (B, T, 1)

        lmk_sum   = (hidden_states * lmk_mask_f).sum(dim=1)                   # (B, D)
        lmk_count = lmk_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)            # (B, 1)
        lmk_mean  = lmk_sum / lmk_count.to(hidden_states.dtype)               # (B, D)

        # ── fallback: mean-pool over non-padding tokens for LMK-less seqs ────
        no_lmk = lmk_mask.sum(dim=1) == 0  # (B,)
        if no_lmk.any():
            attn_f     = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
            attn_sum   = (hidden_states * attn_f).sum(dim=1)
            attn_count = attention_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(hidden_states.dtype)
            lmk_mean[no_lmk] = (attn_sum / attn_count)[no_lmk]

        return lmk_mean

    # ── NEW: unified pool dispatcher ───────────────────────────────────────────
    def _pool(
        self,
        hidden_states: Tensor,            # (B, T, D)
        attention_mask: Tensor,           # (B, T)
        input_ids: Optional[Tensor],      # (B, T) – required only for lmk
    ) -> Tensor:                          # (B, D)
        """Route to the correct pooling function based on args.pooling_source."""
        src = self.args.pooling_source
        if src == 'lmk':
            assert input_ids is not None, "input_ids required for LMK pooling"
            return self._lmk_pool(hidden_states, input_ids, attention_mask)
        elif src == 'mean':
            return mean_pool(hidden_states, attention_mask)
        else:  # 'last' / default
            return last_token_pool(hidden_states, attention_mask)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable()

    def forward(self, query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None,
                kd_labels: Dict[str, Tensor] = None,
                reranker_inps: Dict[str, Tensor] = None):

        if self.args.use_reranker_for_decoder_distillation:
            reranker_kd_labels = self._compute_scores_reranker(reranker_inps)

        if kd_labels is not None:
            kd_labels = kd_labels['kd_labels']
        assert self.args.process_index >= 0

        query_ids = query.pop('query_ids') if 'query_ids' in query else 'no_query_ids'
        if self.trainer.state.global_step < 10:
            with open(self.trainer.args.output_dir + '/model.' + str(os.getpid()), 'at') as pidOut:
                print(self.trainer.state.global_step,
                      'query_ids: ', query['input_ids'].device, query_ids, file=pidOut)

        scores, labels, q_reps, p_reps, all_scores, all_labels = self._compute_scores(query, passage)
        batch_size = q_reps[0].shape[0] if isinstance(q_reps, list) else q_reps.shape[0]

        if not self.args.do_kd_biencoder:
            if isinstance(all_scores, list):
                ce_losses = []
                kd_losses = []
                for scores_layer, labels_layer in zip(all_scores, all_labels):
                    ce = self.cross_entropy(scores_layer, labels_layer)
                    if self.args.use_scaled_loss:
                        ce *= self.args.world_size if self.args.loss_scale <= 0 else self.args.loss_scale
                    ce_losses.append(ce)
                    if self.args.use_reranker_for_decoder_distillation:
                        B = scores_layer.shape[0] // self.args.world_size
                        P = self.args.train_n_passages
                        G = self.args.world_size
                        decoder_logits_for_reranker = collect_blocks_concat_rows(scores_layer, B=B, P=P, G=G)
                        kd = self.cross_entropy(
                            decoder_logits_for_reranker,
                            reranker_kd_labels.view_as(decoder_logits_for_reranker).softmax(dim=-1)
                        )
                        if self.args.use_scaled_loss:
                            kd *= self.args.world_size if self.args.loss_scale <= 0 else self.args.loss_scale
                        kd_losses.append(kd)

                if self.multi_layer_loss_layers is None:
                    self.multi_layer_loss_layers = list(range(1, self.num_layers + 1))

                if self.args.multi_layer_loss_scale == "uniform":
                    multi_layer_loss_scale_tensor = torch.tensor(
                        [1.0] * len(self.multi_layer_loss_layers), dtype=torch.float)
                elif self.args.multi_layer_loss_scale == "layer_scale":
                    raw_weights = torch.tensor(
                        [(l) / self.num_layers for l in self.multi_layer_loss_layers], dtype=torch.float)
                    multi_layer_loss_scale_tensor = raw_weights
                else:
                    multi_layer_loss_scale_tensor = torch.tensor([1.0], dtype=torch.float)

                loss = (torch.stack(ce_losses) * multi_layer_loss_scale_tensor.to(ce_losses[0].device)).mean()
                if self.args.use_reranker_for_decoder_distillation:
                    kd_loss = (torch.stack(kd_losses) * multi_layer_loss_scale_tensor.to(kd_losses[0].device)).mean()
                    loss += kd_loss

                if self.self_distil:
                    layer_id_idx = -1 if self.self_distil_recursive else \
                        self.multi_layer_loss_layers.index(self.self_distil_src_layer)
                    l1_loss_teacher_p_reps = p_reps[layer_id_idx].detach()
                    l1_loss_teacher_q_reps = q_reps[layer_id_idx].detach()
                    l1_losses = []
                    for layer_id in self.self_distil_tgt_layer:
                        idx = self.multi_layer_loss_layers.index(layer_id)
                        l1_losses.append(torch.mean(1.0 - F.cosine_similarity(p_reps[idx], l1_loss_teacher_p_reps, dim=1)))
                        l1_losses.append(torch.mean(1.0 - F.cosine_similarity(q_reps[idx], l1_loss_teacher_q_reps, dim=1)))
                    l1_loss = torch.stack(l1_losses).mean()
                    logger.info(f'Loss: {loss}, l1_loss: {l1_loss}')
                    loss += l1_loss
            else:
                raise TypeError("All scores must be a list")
        else:
            if isinstance(all_scores, list):
                teacher_kd_scores = dist_gather_tensor(kd_labels)
                teacher_kd_target = torch.log_softmax(teacher_kd_scores, dim=-1)
                student_kd_scores = []
                idx = (
                    torch.arange(teacher_kd_target.shape[0]).unsqueeze(1) * self.args.train_n_passages
                    + torch.arange(self.args.train_n_passages).unsqueeze(0)
                )
                for layer_score in all_scores:
                    layer_score = layer_score[torch.arange(layer_score.shape[0]).unsqueeze(1), idx]
                    student_kd_scores.append(torch.log_softmax(layer_score, dim=-1))
                assert student_kd_scores[-1].shape[1] == self.args.train_n_passages

                kd_losses = []
                for student_kd_score in student_kd_scores:
                    kd_losses.append(self.kl_loss_fn(input=student_kd_score, target=teacher_kd_target))

                if self.multi_layer_loss_layers is None:
                    self.multi_layer_loss_layers = list(range(1, self.num_layers + 1))

                if self.args.multi_layer_loss_scale == "uniform":
                    multi_layer_loss_scale_tensor = torch.tensor(
                        [1.0] * len(self.multi_layer_loss_layers), dtype=torch.float)
                elif self.args.multi_layer_loss_scale == "layer_scale":
                    raw_weights = torch.tensor(
                        [(l) / self.num_layers for l in self.multi_layer_loss_layers], dtype=torch.float)
                    multi_layer_loss_scale_tensor = raw_weights
                else:
                    multi_layer_loss_scale_tensor = torch.tensor([1.0], dtype=torch.float)

                loss = (torch.stack(kd_losses) * multi_layer_loss_scale_tensor.to(kd_losses[0].device)).mean()

                if self.args.kd_cont_loss_weight > 0.0:
                    ce_losses = []
                    for scores_layer, labels_layer in zip(all_scores, all_labels):
                        ce = self.cross_entropy(scores_layer, labels_layer)
                        if self.args.use_scaled_loss:
                            ce *= self.args.world_size if self.args.loss_scale <= 0 else self.args.loss_scale
                        ce_losses.append(ce)
                    ce_loss = (torch.stack(ce_losses) * multi_layer_loss_scale_tensor.to(ce_losses[0].device)).mean()
                    loss += self.args.kd_cont_loss_weight * ce_loss

                if self.self_distil:
                    layer_id_idx = -1 if self.self_distil_recursive else \
                        self.multi_layer_loss_layers.index(self.self_distil_src_layer)
                    l1_loss_teacher_p_reps = p_reps[layer_id_idx].detach()
                    l1_loss_teacher_q_reps = q_reps[layer_id_idx].detach()
                    l1_losses = []
                    for layer_id in self.self_distil_tgt_layer:
                        idx = self.multi_layer_loss_layers.index(layer_id)
                        l1_losses.append(torch.mean(1.0 - F.cosine_similarity(p_reps[idx], l1_loss_teacher_p_reps, dim=1)))
                        l1_losses.append(torch.mean(1.0 - F.cosine_similarity(q_reps[idx], l1_loss_teacher_q_reps, dim=1)))
                    l1_loss = torch.stack(l1_losses).mean()
                    loss += l1_loss
                    logger.info(f'Loss: {loss}, l1_loss: {l1_loss}')
            else:
                raise TypeError("All scores must be a list")

        total_n_psg = self.args.world_size * batch_size * self.args.train_n_passages
        return DecoderOutput(
            loss=loss,
            labels=labels[-1].contiguous() if isinstance(labels, list) else labels.contiguous(),
            scores=scores[-1][:, :total_n_psg].contiguous() if isinstance(scores, list)
                   else scores[:, :total_n_psg].contiguous()
        )

    def _compute_scores(self, query: Dict[str, Tensor] = None,
                        passage: Dict[str, Tensor] = None) -> Tuple:
        num_layers = self.num_layers
        layer_indices = (
            list(range(1, num_layers + 1)) if self.multi_layer_loss and self.multi_layer_loss_layers is None
            else self.multi_layer_loss_layers if self.multi_layer_loss
            else [num_layers]
        )

        q_reps_cache = p_reps_cache = None
        if self.cache_minibatch_size is None:
            outputs_q = self.model(**query, output_hidden_states=True)
            outputs_p = self.model(**passage, output_hidden_states=True)
            if self.self_distil_recursive:
                outputs_q_recursive = self.self_distil_recursive_pass(outputs_q.hidden_states[-1], query['attention_mask'])
                outputs_q.hidden_states.append(outputs_q_recursive)
                outputs_p_recursive = self.self_distil_recursive_pass(outputs_p.hidden_states[-1], passage['attention_mask'])
                outputs_p.hidden_states.append(outputs_p_recursive)
        else:
            q_reps_cache, p_reps_cache = self.minibatch_forward(query, passage, layer_indices)

        all_layer_scores  = []
        all_layer_labels  = []
        all_layer_q_reps  = []
        all_layer_p_reps  = []
        all_layer_all_scores = []
        all_layer_all_labels = []

        if self.self_distil_recursive:
            layer_indices.append(num_layers + 1)

        for idx in layer_indices:
            if q_reps_cache is not None and p_reps_cache is not None:
                q_reps = q_reps_cache[idx]
                p_reps = p_reps_cache[idx]
            else:
                # ── CHANGED: use unified _pool() instead of inline if/else ─────
                embeddings_q = self.pooler(
                    self._pool(outputs_q.hidden_states[idx], query['attention_mask'], query.get('input_ids'))
                )
                embeddings_p = self.pooler(
                    self._pool(outputs_p.hidden_states[idx], passage['attention_mask'], passage.get('input_ids'))
                )
                q_reps = F.normalize(embeddings_q, p=2, dim=-1)
                p_reps = F.normalize(embeddings_p, p=2, dim=-1)

            all_q_reps = dist_gather_tensor(q_reps)
            all_p_reps = dist_gather_tensor(p_reps)

            assert all_p_reps.shape[0] == self.args.world_size * q_reps.shape[0] * self.args.train_n_passages, \
                f'p reps shape {all_p_reps.shape} q reps shape {q_reps.shape}'

            if self.trainer.state.global_step < 10:
                with open(self.trainer.args.output_dir + '/model.' + str(os.getpid()), 'at') as pidOut:
                    print('merging: ', self.args.world_size,
                          str(q_reps.device), str(q_reps.shape),
                          str(p_reps.device), str(p_reps.shape),
                          str(all_q_reps.device), str(all_q_reps.shape),
                          str(all_p_reps.device), str(all_p_reps.shape), file=pidOut)

            all_scores, all_labels = full_contrastive_scores_and_labels(
                query=all_q_reps, key=all_p_reps,
                use_all_pairs=self.args.full_contrastive_loss)

            if self.args.l2_normalize:
                if self.args.t_warmup:
                    scale = 1 / self.args.t * min(1.0, self.trainer.state.global_step / self.args.warmup_steps)
                    scale = max(1.0, scale)
                else:
                    scale = 1 / self.args.t
                all_scores = all_scores * scale

            start = self.args.process_index * q_reps.shape[0]
            local_query_indices = torch.arange(start, start + q_reps.shape[0], dtype=torch.long).to(q_reps.device)

            scores = all_scores.index_select(dim=0, index=local_query_indices)
            labels = all_labels.index_select(dim=0, index=local_query_indices)

            all_layer_scores.append(scores)
            all_layer_labels.append(labels)
            all_layer_q_reps.append(q_reps)
            all_layer_p_reps.append(p_reps)
            all_layer_all_scores.append(all_scores)
            all_layer_all_labels.append(all_labels)

        return all_layer_scores, all_layer_labels, all_layer_q_reps, all_layer_p_reps, \
               all_layer_all_scores, all_layer_all_labels

    def self_distil_recursive_pass(self, reps, attn_mask, itr=1):
        cache_position = torch.arange(0, reps.shape[1], device=reps.device)
        position_ids = cache_position.unsqueeze(0)
        mask = create_causal_mask(
            config=self.model.config,
            input_embeds=reps,
            attention_mask=attn_mask,
            cache_position=cache_position,
            past_key_values=None
        )
        position_embeddings = self.model.model.model.rotary_emb(reps, position_ids)
        hidden_states = reps
        for _ in range(itr):
            for decoder_layer in self.model.model.model.layers[:self.model.config.num_hidden_layers]:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs
            hidden_states = self.model.model.model.norm(hidden_states)
        return hidden_states

    def minibatch_forward(self, query, passage, layer_indices):
        mini_batch_size_query = self.cache_minibatch_size
        mini_batch_size_docs  = self.document_cache_minibatch_size
        total_batch_size      = query['input_ids'].size(0)
        train_n_passages      = self.args.train_n_passages
        use_lmk               = (self.args.pooling_source == 'lmk')

        hidden_states_q_all = []
        hidden_states_p_all = []

        for i in range(0, total_batch_size, mini_batch_size_query):
            q_batch = {
                'input_ids':      query['input_ids'][i:i + mini_batch_size_query],
                'attention_mask': query['attention_mask'][i:i + mini_batch_size_query],
            }

            if mini_batch_size_docs is not None:
                outputs_q = self.model(**q_batch, output_hidden_states=True)
                # ── CHANGED: use _pool with input_ids for LMK ─────────────────
                q_hidden_states = [
                    F.normalize(
                        self.pooler(self._pool(x, q_batch['attention_mask'],
                                               q_batch['input_ids'] if use_lmk else None)),
                        p=2, dim=1
                    ) if layer_i in layer_indices else torch.tensor([])
                    for layer_i, x in enumerate(outputs_q.hidden_states)
                ]
                hidden_states_q_all.append(q_hidden_states)
                final_hidden_states = outputs_q.hidden_states[-1]
                del outputs_q

                if self.self_distil_recursive:
                    outputs_q_recursive = self.self_distil_recursive_pass(final_hidden_states, q_batch['attention_mask'])
                    q_hs_recursive = F.normalize(
                        self.pooler(self._pool(outputs_q_recursive, q_batch['attention_mask'],
                                               q_batch['input_ids'] if use_lmk else None)),
                        p=2, dim=1
                    )
                    hidden_states_q_all[-1].append(q_hs_recursive)
                    del outputs_q_recursive

                start_idx = i * train_n_passages
                end_idx   = (i + mini_batch_size_query) * train_n_passages
                for j in range(start_idx, end_idx, mini_batch_size_docs):
                    sl = slice(j, min(end_idx, j + mini_batch_size_docs))
                    p_batch = {
                        'input_ids':      passage['input_ids'][sl],
                        'attention_mask': passage['attention_mask'][sl],
                    }
                    outputs_p = self.model(**p_batch, output_hidden_states=True)
                    p_hidden_states = [
                        F.normalize(
                            self.pooler(self._pool(x, p_batch['attention_mask'],
                                                   p_batch['input_ids'] if use_lmk else None)),
                            p=2, dim=1
                        ) if layer_i in layer_indices else torch.tensor([])
                        for layer_i, x in enumerate(outputs_p.hidden_states)
                    ]
                    hidden_states_p_all.append(p_hidden_states)
                    final_hidden_states = outputs_p.hidden_states[-1]
                    del outputs_p

                    if self.self_distil_recursive:
                        outputs_p_recursive = self.self_distil_recursive_pass(final_hidden_states, p_batch['attention_mask'])
                        p_hs_recursive = F.normalize(
                            self.pooler(self._pool(outputs_p_recursive, p_batch['attention_mask'],
                                                   p_batch['input_ids'] if use_lmk else None)),
                            p=2, dim=1
                        )
                        hidden_states_p_all[-1].append(p_hs_recursive)
                        del outputs_p_recursive
            else:
                start_idx = i * train_n_passages
                end_idx   = (i + mini_batch_size_query) * train_n_passages
                p_batch = {
                    'input_ids':      passage['input_ids'][start_idx:end_idx],
                    'attention_mask': passage['attention_mask'][start_idx:end_idx],
                }

                outputs_q = self.model(**q_batch, output_hidden_states=True)
                q_hidden_states = [
                    F.normalize(
                        self.pooler(self._pool(x, q_batch['attention_mask'],
                                               q_batch['input_ids'] if use_lmk else None)),
                        p=2, dim=1
                    ) if layer_i in layer_indices else torch.tensor([])
                    for layer_i, x in enumerate(outputs_q.hidden_states)
                ]
                hidden_states_q_all.append(q_hidden_states)
                final_hidden_states = outputs_q.hidden_states[-1]
                del outputs_q

                if self.self_distil_recursive:
                    outputs_q_recursive = self.self_distil_recursive_pass(final_hidden_states, q_batch['attention_mask'])
                    q_hs_recursive = F.normalize(
                        self.pooler(self._pool(outputs_q_recursive, q_batch['attention_mask'],
                                               q_batch['input_ids'] if use_lmk else None)),
                        p=2, dim=1
                    )
                    hidden_states_q_all[-1].append(q_hs_recursive)
                    del outputs_q_recursive

                outputs_p = self.model(**p_batch, output_hidden_states=True)
                p_hidden_states = [
                    F.normalize(
                        self.pooler(self._pool(x, p_batch['attention_mask'],
                                               p_batch['input_ids'] if use_lmk else None)),
                        p=2, dim=1
                    ) if layer_i in layer_indices else torch.tensor([])
                    for layer_i, x in enumerate(outputs_p.hidden_states)
                ]
                hidden_states_p_all.append(p_hidden_states)
                final_hidden_states = outputs_p.hidden_states[-1]
                del outputs_p

                if self.self_distil_recursive:
                    outputs_p_recursive = self.self_distil_recursive_pass(final_hidden_states, p_batch['attention_mask'])
                    p_hs_recursive = F.normalize(
                        self.pooler(self._pool(outputs_p_recursive, p_batch['attention_mask'],
                                               p_batch['input_ids'] if use_lmk else None)),
                        p=2, dim=1
                    )
                    hidden_states_p_all[-1].append(p_hs_recursive)
                    del outputs_p_recursive

        num_layers_stored = len(hidden_states_q_all[0])
        final_hidden_states_q = [
            torch.cat([batch[layer] for batch in hidden_states_q_all], dim=0)
            for layer in range(num_layers_stored)
        ]
        final_hidden_states_p = [
            torch.cat([batch[layer] for batch in hidden_states_p_all], dim=0)
            for layer in range(num_layers_stored)
        ]
        return final_hidden_states_q, final_hidden_states_p

    def _compute_scores_reranker(self, reranker_inps: Dict[str, Tensor] = None) -> Tuple:
        with torch.no_grad():
            end_idx = len(reranker_inps['input_ids'])
            mini_batch_size_reranker = 256
            outputs_reranker = []
            for i in range(0, end_idx, mini_batch_size_reranker):
                sub_batch = {
                    'input_ids':      reranker_inps['input_ids'][i:min(end_idx, i + mini_batch_size_reranker)],
                    'attention_mask': reranker_inps['attention_mask'][i:min(end_idx, i + mini_batch_size_reranker)],
                }
                outputs_reranker.append(
                    self.reranker_model(**sub_batch, return_dict=True).logits.squeeze()
                )
            outputs_reranker = torch.cat(outputs_reranker, dim=0)
        return dist_gather_tensor(outputs_reranker)

    def save(self, output_dir: str):
        self.model.save_pretrained(output_dir, safe_serialization=False)
        if self.args.add_pooler:
            torch.save(self.pooler.state_dict(), os.path.join(output_dir, 'pooler.pt'))