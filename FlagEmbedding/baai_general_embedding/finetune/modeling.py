import logging
from einops import rearrange, repeat
from dataclasses import dataclass
from typing import Dict, Optional
import os
import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.nn.attention import SDPBackend
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.modernbert.modeling_modernbert import ModernBertRotaryEmbedding
from .infonce import InfoNCE
import pdb
import numpy as np
from .utils import full_contrastive_scores_and_labels, LatentAttentionConfig, print_gpu_memory
torch.use_deterministic_algorithms(True)
logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None



# helper classes borrowed for latent attention
# LATENT_ATTENTION_TYPE = "latent_attention"
# AutoConfig.register(LATENT_ATTENTION_TYPE, LatentAttentionConfig)
# LatentAttentionConfig.register_for_auto_class()

class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.LayerNorm(dim)
        self.norm_context = torch.nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)
        return self.fn(x, **kwargs)

class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * torch.nn.functional.gelu(gates)

class FeedForward(torch.nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            torch.nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class Attention(torch.nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = torch.nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = torch.nn.Linear(inner_dim, query_dim, bias = False)

    def forward(self, x, context = None, mask = None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        with torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION,SDPBackend.MATH], set_priority=False):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class LatentAttentionModel(PreTrainedModel):
    config_class = LatentAttentionConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: LatentAttentionConfig):
        super().__init__(config)
        ## cross-attention block
        num_latents, latent_dim, cross_heads, cross_dim_head = config.num_latents_value, config.latent_dim, config.num_cross_heads, config.cross_dim_head
        dim = config.hidden_dim
        # init latent_attention and latents
        self.cross_attend_blocks = torch.nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head),
                    context_dim = dim),
            PreNorm(latent_dim, FeedForward(latent_dim)),
        ])
        self.output_normalize = config.output_normalize
        self.register_parameter("latents", torch.nn.Parameter(torch.randn(num_latents, latent_dim)))

    def forward(self, hiddens, attention_mask: torch.Tensor=None):
        ## cross-attention block
        cross_attn, cross_ff = self.cross_attend_blocks
        b, *_, device = *hiddens.shape, hiddens.device
        # x = repeat(self.latents, 'n d -> b n d', b = b)
        x = self.latents.unsqueeze(0).expand(b, -1, -1)
        
        def attn_fn(h, x):
            return cross_attn(h, context=x)

        def ff_fn(h):
            return cross_ff(h)

        if self.gradient_checkpointing and self.training:
            hiddens = checkpoint(attn_fn, hiddens, x) + hiddens
            hiddens = checkpoint(ff_fn, hiddens) + hiddens
        else:
            hiddens = attn_fn(hiddens, x) + hiddens
            hiddens = ff_fn(hiddens) + hiddens

        if attention_mask !=None:
            s = torch.sum(hiddens * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            hiddens = s / d
            if self.output_normalize:
                hiddens = torch.nn.functional.normalize(hiddens, p=2, dim=-1)
        return hiddens
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LatentAttentionModel):
            module.gradient_checkpointing = value


class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 infonce: bool = False,
                 full_contrastive = False,
                 lora = False,
                 disable_rope = False
                 ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if disable_rope:
            logger.info(f'Disabling rope for this model')
            def disable_rope(model):
                with torch.no_grad():
                    for m in model.modules():
                        if isinstance(m, ModernBertRotaryEmbedding):
                            m.inv_freq.zero_()
                            m.attention_scaling = 1.0
            disable_rope(self.model) # this will disable rope embeddings as an ablation to test cls extrapolation
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.loss_fun = InfoNCE(temperature=temperature)
        self.infonce = infonce
        self.full_contrastive = full_contrastive
        self.lora = lora
        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        if self.sentence_pooling_method == 'latent_attn':
            self.num_latents = 64
            hidden_size = self.model.config.hidden_size
            self.latent_attention_config = LatentAttentionConfig(
                num_latents_value=self.num_latents, # we use fix latent dim as 64 for now, based on this the number of heads may vary.
                num_cross_heads=hidden_size//self.num_latents,
                output_normalize=False, #since we are manually doing it in encode func.
                hidden_dim=hidden_size,
                latent_dim=hidden_size,
                cross_dim_head=hidden_size
            )
            self.latent_attention_model = LatentAttentionModel(self.latent_attention_config)
            logger.info(f'Initialized latent attn model.')
        self.temperature = temperature
        self._keys_to_ignore_on_save = None
        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self,**kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)
        if self.sentence_pooling_method == 'latent_attn':
            self.latent_attention_model.gradient_checkpointing_enable(**kwargs)

    def mean_pool_on_lmk(self,
                    token_embeddings: torch.Tensor,
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor,
                    lmk_token_id: int) -> torch.Tensor:
        """
        Returns: mean pooled tensor of shape (B, D)
        """
        # boolean mask where token is [LMK] and not padding: (B, Seq)
        lmk_mask = (input_ids == lmk_token_id) & (attention_mask == 1)

        # float mask for multiplication: (B, Seq, 1)
        lmk_mask_f = lmk_mask.to(dtype=token_embeddings.dtype).unsqueeze(-1)

        # sum of [LMK] embeddings per example: (B, D)
        lmk_sum = (token_embeddings * lmk_mask_f).sum(dim=1)

        # number of [LMK] tokens per example: (B, 1)
        lmk_count = lmk_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(token_embeddings.dtype)

        # mean over [LMK] positions (if none, we will divide by 1 -> but result would be 0)
        lmk_mean = lmk_sum / lmk_count

        # For examples that had zero [LMK] tokens, fallback to mean over non-padded tokens:
        no_lmk_examples = (lmk_mask.sum(dim=1) == 0)  # (B,)
        if no_lmk_examples.any():
            # compute mean over attention_mask==1 tokens: (B, D)
            attn_mask_f = attention_mask.to(dtype=token_embeddings.dtype).unsqueeze(-1)
            attn_sum = (token_embeddings * attn_mask_f).sum(dim=1)
            attn_count = attention_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(token_embeddings.dtype)
            attn_mean = attn_sum / attn_count

            # replace rows where no [LMK] with attn_mean
            lmk_mean[no_lmk_examples] = attn_mean[no_lmk_examples]

        return lmk_mean  # shape (B, D)
    
    def mean_pool_on_every_k_tokens(self,
                    token_embeddings: torch.Tensor,
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor,
                    k: int = 64 # position to mean pool on every k
                    ) -> torch.Tensor:
        """
        Returns: mean pooled tensor of shape (B, D)
        """
        # boolean mask where token is [LMK] and not padding: (B, Seq)
        positions = torch.arange(0, input_ids.shape[1]).to(input_ids.device).unsqueeze(0)

        lmk_mask = ((positions%k) == 0) & (attention_mask == 1)

        # float mask for multiplication: (B, Seq, 1)
        lmk_mask_f = lmk_mask.to(dtype=token_embeddings.dtype).unsqueeze(-1)

        # sum of [LMK] embeddings per example: (B, D)
        lmk_sum = (token_embeddings * lmk_mask_f).sum(dim=1)

        # number of [LMK] tokens per example: (B, 1)
        lmk_count = lmk_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(token_embeddings.dtype)

        # mean over [LMK] positions (if none, we will divide by 1 -> but result would be 0)
        lmk_mean = lmk_sum / lmk_count

        # For examples that had zero [LMK] tokens, fallback to mean over non-padded tokens:
        no_lmk_examples = (lmk_mask.sum(dim=1) == 0)  # (B,)
        if no_lmk_examples.any():
            # compute mean over attention_mask==1 tokens: (B, D)
            attn_mask_f = attention_mask.to(dtype=token_embeddings.dtype).unsqueeze(-1)
            attn_sum = (token_embeddings * attn_mask_f).sum(dim=1)
            attn_count = attention_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(token_embeddings.dtype)
            attn_mean = attn_sum / attn_count

            # replace rows where no [LMK] with attn_mean
            lmk_mean[no_lmk_examples] = attn_mean[no_lmk_examples]

        return lmk_mean  # shape (B, D)

    def sentence_embedding(self, hidden_state, input_ids, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        elif 'lmk' in self.sentence_pooling_method:
            assert self.tokenizer.sep_token_id != None
            return self.mean_pool_on_lmk(
                token_embeddings=hidden_state,
                input_ids=input_ids,
                attention_mask=mask,
                lmk_token_id=self.tokenizer.sep_token_id
                )
        elif 'every_64' == self.sentence_pooling_method:
            return self.mean_pool_on_every_k_tokens(
                token_embeddings=hidden_state,
                input_ids=input_ids,
                attention_mask=mask,
                k=64
            )
        elif self.sentence_pooling_method == 'latent_attn':
            outs = self.latent_attention_model(hiddens=hidden_state, attention_mask=mask)
            return outs

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['input_ids'], features['attention_mask'])
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            # q_reps = bs, dim
            # p_reps = bs * grp_size, dim
            # out = bs, dim @ dim , bs * grp_size
            # out = bs, bs * grp_size
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))


    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None, weight: Tensor=None):
        batch_size = query['input_ids'].shape[0]
        group_size = passage['input_ids'].shape[0] // batch_size
        q_reps = self.encode(query)
        #p_reps = self.encode(passage)
        # sub batching for training with larger bs when a lot of negatives are present
        p_reps = []
        for i in range(group_size):
            sub_batch = {k:v[(i)*batch_size:(i+1)*batch_size] for k,v in passage.items()}
            p_reps.append(self.encode(sub_batch))
        p_reps = torch.cat(p_reps, dim=0)
        
        if self.training:
            #pdb.set_trace()
            if self.negatives_cross_device: 
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                if teacher_score != None:
                    teacher_score = self._dist_gather_tensor(teacher_score.squeeze())
                    assert len(teacher_score.shape) == 1
            if teacher_score!=None:
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores / self.temperature
                scores = scores.view(q_reps.size(0), -1) # bs, bs*grp_size
                teacher_score = teacher_score.view(q_reps.size(0), -1).detach() # bs, grp_size
                teacher_targets = torch.nn.functional.softmax(teacher_score, dim=-1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                group_size = p_reps.size(0) // q_reps.size(0)
                target = target * group_size
                loss = 0
                # bge m3 kd loss
                mask = torch.zeros_like(scores)
                for i in range(group_size):
                    temp_target = target + i # bs -> [0, grp_size, grp_size*2, .... grp_size*(bs-1)] + i
                    temp_scores = scores + mask # bs, bs*grp_size
                    temp_loss = torch.nn.functional.cross_entropy(temp_scores, temp_target, reduction="none")  # bs
                    if weight is not None and weight.numel()>0: #bge
                        weight = weight.view(-1,1)
                        weight = self._dist_gather_tensor(weight)
                        loss += torch.mean(teacher_targets[:, i] * temp_loss * weight)
                    else:
                        loss += torch.mean(teacher_targets[:, i] * temp_loss)
                    mask = torch.scatter(mask, dim=-1, index=temp_target.unsqueeze(-1),
                                        value=torch.finfo(scores.dtype).min)
            elif self.full_contrastive:
                scores,labels = full_contrastive_scores_and_labels(q_reps,p_reps)
                scores = scores / self.temperature
                loss = self.compute_loss(scores, labels) 

                loss *= self.world_size
            else:
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores / self.temperature
                scores = scores.view(q_reps.size(0), -1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * (p_reps.size(0) // q_reps.size(0))
                    
                if weight is not None and weight.numel()>0:
                    weight = weight.view(-1,1)
                    weight = self._dist_gather_tensor(weight)
                    loss = torch.nn.functional.cross_entropy(scores, target, reduction='none')
                    loss = torch.mean(loss * weight.view(-1,1))
                else:
                    loss = self.compute_loss(scores, target)   
                loss *= self.world_size
        else:
            if teacher_score!=None:
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores / self.temperature
                scores = scores.view(q_reps.size(0), -1) # bs, bs*grp_size
                teacher_score = teacher_score.view(q_reps.size(0), -1).detach() # bs, grp_size
                teacher_targets = torch.nn.functional.softmax(teacher_score, dim=-1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                group_size = p_reps.size(0) // q_reps.size(0)
                target = target * group_size
                loss = 0
                # bge m3 kd loss
                mask = torch.zeros_like(scores)
                for i in range(group_size):
                    temp_target = target + i # bs -> [0, grp_size, grp_size*2, .... grp_size*(bs-1)] + i
                    temp_scores = scores + mask # bs, bs*grp_size
                    temp_loss = torch.nn.functional.cross_entropy(temp_scores, temp_target, reduction="none")  # bs
                    if weight is not None and weight.numel()>0: #bge
                        weight = weight.view(-1,1)
                        weight = self._dist_gather_tensor(weight)
                        loss += torch.mean(teacher_targets[:, i] * temp_loss * weight)
                    else:
                        loss += torch.mean(teacher_targets[:, i] * temp_loss)
                    mask = torch.scatter(mask, dim=-1, index=temp_target.unsqueeze(-1),
                                        value=torch.finfo(scores.dtype).min)
                loss *= self.world_size
            elif self.full_contrastive:
                scores,labels = full_contrastive_scores_and_labels(q_reps,p_reps)
                scores = scores / self.temperature
                loss = self.compute_loss(scores, labels) 

                loss *= self.world_size
            else:
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores.view(q_reps.size(0), -1)

                scores = scores / self.temperature
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * (p_reps.size(0) // q_reps.size(0))
                    
                if weight is not None and weight.numel()>0:
                    weight = weight.view(-1,1)
                    weight = self._dist_gather_tensor(weight)
                    loss = torch.nn.functional.cross_entropy(scores, target, reduction='none')
                    loss = torch.mean(loss * weight.view(-1,1))
                else:
                    loss = self.compute_loss(scores, target) 
                loss *= self.world_size
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
    

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.sentence_pooling_method=='latent_attn':
            save_dir = os.path.join(output_dir, "latent_attn_model")
            self.latent_attention_model.save_pretrained(save_dir)
    def load(self,checkpoint_path):
        return self.model.from_pretrained(checkpoint_path)
