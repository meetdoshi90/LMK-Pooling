import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from torch import Tensor
from transformers import (
    AutoModel,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput

from config import Arguments
from logger_config import logger
from utils import dist_gather_tensor, select_grouped_indices, full_contrastive_scores_and_labels, last_token_pool, full_contrastive_scores_and_labels_kd, mean_pool


@dataclass
class BiencoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiencoderModel(nn.Module):
    def __init__(self, args: Arguments,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.args = args
        self.pooler = nn.Linear(self.lm_q.config.hidden_size, args.out_dimension) if args.add_pooler else nn.Identity()

        from trainers import BiencoderTrainer
        self.trainer: Optional[BiencoderTrainer] = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
         self.lm_p.gradient_checkpointing_enable()
         self.lm_q.gradient_checkpointing_enable()

    def forward(self, query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None):
        assert self.args.process_index >= 0

        query_ids=query.pop('query_ids') if 'query_ids' in query else 'no_query_ids'
        if self.trainer.state.global_step < 10:
            with open(self.trainer.args.output_dir+'/model.'+str(os.getpid()), 'at') as pidOut:
                print(self.trainer.state.global_step, 
                    'query_ids: ', query['input_ids'].device, query_ids, file=pidOut)


        scores, labels, q_reps, p_reps, all_scores, all_labels = self._compute_scores(query, passage)

        start = self.args.process_index * q_reps.shape[0]
        group_indices = select_grouped_indices(scores=scores,
                                               group_size=self.args.train_n_passages,
                                               start=start * self.args.train_n_passages)

        if not self.args.do_kd_biencoder:
            # training biencoder from scratch
            if self.args.use_scaled_loss:
                loss = self.cross_entropy(all_scores, all_labels)
                loss *= self.args.world_size if self.args.loss_scale <= 0 else self.args.loss_scale
            else:
                loss = self.cross_entropy(scores, labels)
        else:
            # training biencoder with kd
            # batch_size x train_n_passage
            group_scores = torch.gather(input=scores, dim=1, index=group_indices)
            assert group_scores.shape[1] == self.args.train_n_passages
            group_log_scores = torch.log_softmax(group_scores, dim=-1)
            kd_log_target = torch.log_softmax(query['kd_labels'], dim=-1)

            kd_loss = self.kl_loss_fn(input=group_log_scores, target=kd_log_target)

            # (optionally) mask out hard negatives
            if self.training and self.args.kd_mask_hn:
                scores = torch.scatter(input=scores, dim=1, index=group_indices[:, 1:], value=float('-inf'))
            if self.args.use_scaled_loss:
                ce_loss = self.cross_entropy(all_scores, all_labels)
                ce_loss *= self.args.world_size if self.args.loss_scale <= 0 else self.args.loss_scale
            else:
                ce_loss = self.cross_entropy(scores, labels)

            loss = self.args.kd_cont_loss_weight * ce_loss + kd_loss

        total_n_psg = self.args.world_size * q_reps.shape[0] * self.args.train_n_passages

        return BiencoderOutput(loss=loss, q_reps=q_reps, p_reps=p_reps,
                               labels=labels.contiguous(),
                               scores=scores[:, :total_n_psg].contiguous())

    def _compute_scores(self, query: Dict[str, Tensor] = None,
                        passage: Dict[str, Tensor] = None) -> Tuple:
        q_reps = self._encode(self.lm_q, query)
        p_reps = self._encode(self.lm_p, passage)


        all_q_reps = dist_gather_tensor(q_reps)
        all_p_reps = dist_gather_tensor(p_reps)
        assert all_p_reps.shape[0] == self.args.world_size * q_reps.shape[0] * self.args.train_n_passages

        if self.trainer.state.global_step < 10:
            with open(self.trainer.args.output_dir+'/model.'+str(os.getpid()), 'at') as pidOut:
                print(  'merging: ', self.args.world_size,
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
        # batch_size x (world_size x batch_size x train_n_passage)
        scores = all_scores.index_select(dim=0, index=local_query_indices)
        labels = all_labels.index_select(dim=0, index=local_query_indices)

        return scores, labels, q_reps, p_reps, all_scores, all_labels

    def _encode(self, encoder: PreTrainedModel, input_dict: dict) -> Optional[torch.Tensor]:
        if not input_dict:
            return None
        outputs = encoder(**{k: v for k, v in input_dict.items() if k not in ['kd_labels']}, return_dict=True)
        hidden_state = outputs.last_hidden_state

        # hidden_state.shape is (bs, seqlen, hidden_dim)
        if self.args.pooling_source=='cls':
            embeds = hidden_state[:, 0]
        elif self.args.pooling_source=='mean':
            # code borrowed from SentenceTransformers Pooling.py
            # to account for the mask when taking the mean
            attention_mask = input_dict['attention_mask'] # (bs,seqlen)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).type(hidden_state.dtype)  # (bs, seqlen, hiddendim)
            sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)   # (bs, hidden_dim)
            sum_mask = input_mask_expanded.sum(1)  # (bs, hidden_dim)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            embeds = sum_embeddings / sum_mask  # (bs, hidden_dim)
        embeds = self.pooler(embeds)
        if self.args.l2_normalize:
            embeds = F.normalize(embeds, dim=-1)
        return embeds.contiguous()

    @classmethod
    def build(cls, args: Arguments, **hf_kwargs):
        # load local
        if os.path.isdir(args.model_name_or_path):
            if not args.share_encoder:
                _qry_model_path = os.path.join(args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = args.model_name_or_path
                    _psg_model_path = args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = AutoModel.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModel.from_pretrained(_psg_model_path, **hf_kwargs)
            else:
                logger.info(f'loading shared model weight from {args.model_name_or_path}')
                lm_q = AutoModel.from_pretrained(args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = AutoModel.from_pretrained(args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if not args.share_encoder else lm_q

        model = cls(args=args, lm_q=lm_q, lm_p=lm_p)
        return model

    def save(self, output_dir: str):
        if not self.args.share_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'passage_model'), exist_ok=True)
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
        else:
            self.lm_q.save_pretrained(output_dir, safe_serialization=False) #backward compatibility for evaluation
        if self.args.add_pooler:
            torch.save(self.pooler.state_dict(), os.path.join(output_dir, 'pooler.pt'))

class BiencoderModelKD(BiencoderModel):
    '''
    combine both student and teacher model in this class to work with deepspeed
    '''
    def __init__(self, args: Arguments,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 teacher: PreTrainedModel= None,):
        super().__init__(args, lm_p, lm_q)
        self.teacher_model = teacher
        self.teacher_model.eval()
        self.l1_loss = torch.nn.L1Loss()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.lm_p.gradient_checkpointing_enable()
        self.lm_q.gradient_checkpointing_enable()
        self.teacher_model.gradient_checkpointing_enable()

    def forward(self, query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None,
                query_teacher: Dict[str, Tensor] = None,
                passage_teacher:Dict[str, Tensor] = None,):
        assert self.args.process_index >= 0

        query_ids=query.pop('query_ids') if 'query_ids' in query else 'no_query_ids'
        if self.trainer.state.global_step < 10:
            with open(self.trainer.args.output_dir+'/model.'+str(os.getpid()), 'at') as pidOut:
                print(self.trainer.state.global_step, 
                    'query_ids: ', query['input_ids'].device, query_ids, file=pidOut)


        scores, labels, q_reps, p_reps, all_scores, all_labels = self._compute_scores(query, passage) 
        scores_teacher, _, _, _, all_scores_teacher, _ = self._compute_scores(query_teacher, passage_teacher, is_teacher=True)

        start = self.args.process_index * q_reps.shape[0]
        group_indices = select_grouped_indices(scores=scores,
                                               group_size=self.args.train_n_passages,
                                               start=start * self.args.train_n_passages)

        if self.args.direct_similarity_distill:
            loss = self.l1_loss(all_scores, all_scores_teacher)
        else:
            if query_ids[0][:2] in ['24']: #for fever, use hard labels
                loss = 0.5 * self.cross_entropy(all_scores, all_labels)
            else:
                soft_target = all_scores_teacher.softmax(dim=-1)
                loss = self.cross_entropy(all_scores, soft_target)

        total_n_psg = self.args.world_size * q_reps.shape[0] * self.args.train_n_passages

        return BiencoderOutput(loss=loss, q_reps=q_reps, p_reps=p_reps,
                               labels=labels.contiguous(),
                               scores=scores[:, :total_n_psg].contiguous())

    def _compute_scores(self, query: Dict[str, Tensor] = None,
                        passage: Dict[str, Tensor] = None,
                        is_teacher=False) -> Tuple:

        if is_teacher:
            with torch.no_grad():
                teacher_outputs_q = self.teacher_model(**query)
                teacher_outputs_p = self.teacher_model(**passage)

                if self.args.teacher_pooling_source == 'mean':
                    embeddings_q = mean_pool(teacher_outputs_q['last_hidden_state'], query['attention_mask'])
                    embeddings_p = mean_pool(teacher_outputs_p['last_hidden_state'], passage['attention_mask'])
                else:
                    embeddings_q = last_token_pool(teacher_outputs_q['last_hidden_state'], query['attention_mask'])
                    embeddings_p = last_token_pool(teacher_outputs_p['last_hidden_state'], passage['attention_mask'])
                # normalize embeddings
                q_reps = F.normalize(embeddings_q, p=2, dim=-1)
                p_reps = F.normalize(embeddings_p, p=2, dim=-1)
        else:
            q_reps = self._encode(self.lm_q, query)
            p_reps = self._encode(self.lm_p, passage)


        all_q_reps = dist_gather_tensor(q_reps)
        all_p_reps = dist_gather_tensor(p_reps)
        assert all_p_reps.shape[0] == self.args.world_size * q_reps.shape[0] * self.args.train_n_passages

        if self.trainer.state.global_step < 10:
            with open(self.trainer.args.output_dir+'/model.'+str(os.getpid()), 'at') as pidOut:
                print(  'merging: ', self.args.world_size,
                        str(q_reps.device), str(q_reps.shape),
                        str(p_reps.device), str(p_reps.shape),
                        str(all_q_reps.device), str(all_q_reps.shape),
                        str(all_p_reps.device), str(all_p_reps.shape), file=pidOut)

        all_scores, all_labels = full_contrastive_scores_and_labels_kd(
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
        # batch_size x (world_size x batch_size x train_n_passage)
        scores = all_scores.index_select(dim=0, index=local_query_indices)
        labels = all_labels.index_select(dim=0, index=local_query_indices)

        return scores, labels, q_reps, p_reps, all_scores, all_labels
    @classmethod
    def build(cls, args: Arguments, **hf_kwargs):
        # load local
        if os.path.isdir(args.model_name_or_path):
            if not args.share_encoder:
                _qry_model_path = os.path.join(args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = args.model_name_or_path
                    _psg_model_path = args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = AutoModel.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModel.from_pretrained(_psg_model_path, **hf_kwargs)
            else:
                logger.info(f'loading shared model weight from {args.model_name_or_path}')
                lm_q = AutoModel.from_pretrained(args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = AutoModel.from_pretrained(args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if not args.share_encoder else lm_q

        teacher = AutoModel.from_pretrained(args.teacher)
        model = cls(args=args, lm_q=lm_q, lm_p=lm_p, teacher=teacher)
        return model

class BiencoderModelForInference(BiencoderModel):
    def __init__(self, args: Arguments,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel):
        nn.Module.__init__(self)
        self.args = args
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = nn.Linear(self.lm_q.config.hidden_size, args.out_dimension) if args.add_pooler else nn.Identity()

    @torch.no_grad()
    def forward(self, query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None):
        q_reps = self._encode(self.lm_q, query)
        p_reps = self._encode(self.lm_p, passage)
        return BiencoderOutput(q_reps=q_reps, p_reps=p_reps)

    @classmethod
    def build(cls, args: Arguments, **hf_kwargs):
        model_name_or_path = args.model_name_or_path

        # load local
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            if os.path.exists(_qry_model_path):
                logger.info(f'found separate weight for query/passage encoders')
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = AutoModel.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModel.from_pretrained(_psg_model_path, **hf_kwargs)
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                lm_q = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else:
            logger.info(f'try loading tied weight {model_name_or_path}')
            lm_q = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        model = cls(args=args, lm_q=lm_q, lm_p=lm_p)

        pooler_path = os.path.join(args.model_name_or_path, 'pooler.pt')
        if os.path.exists(pooler_path):
            logger.info('loading pooler weights from local files')
            state_dict = torch.load(pooler_path, map_location="cpu")
            model.pooler.load_state_dict(state_dict)
        else:
            assert not args.add_pooler
            logger.info('No pooler will be loaded')
        return model
