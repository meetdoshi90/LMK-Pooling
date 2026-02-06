import os
import torch
from torch.nn.attention import SDPBackend
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from transformers import AutoModel, AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from sentence_transformers import models

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

    def forward(self, sentence_embedding, attention_mask, **features):
        ## cross-attention block
        cross_attn, cross_ff = self.cross_attend_blocks
        b, *_, device = *sentence_embedding.shape, sentence_embedding.device
        # x = repeat(self.latents, 'n d -> b n d', b = b)
        x = self.latents.unsqueeze(0).expand(b, -1, -1)
        
        def attn_fn(h, x):
            return cross_attn(h, context=x)

        def ff_fn(h):
            return cross_ff(h)
        
        if self.is_gradient_checkpointing and self.training:
            sentence_embedding = checkpoint(attn_fn, sentence_embedding, x) + sentence_embedding
            sentence_embedding = checkpoint(ff_fn, sentence_embedding) + sentence_embedding
        else:
            sentence_embedding = attn_fn(sentence_embedding, x) + sentence_embedding
            sentence_embedding = ff_fn(sentence_embedding) + sentence_embedding

        if attention_mask !=None:
            s = torch.sum(sentence_embedding * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            sentence_embedding = s / d
            if self.output_normalize:
                sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=-1)
        return (sentence_embedding,)
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LatentAttentionModel):
            module.gradient_checkpointing = value

if __name__ == "__main__":
    # Test loading a latent attn model
    LATENT_ATTENTION_TYPE = "latent_attention"
    AutoConfig.register(LATENT_ATTENTION_TYPE, LatentAttentionConfig)
    AutoModel.register(LatentAttentionConfig, LatentAttentionModel)
    LatentAttentionModel.register_for_auto_class()

    model_path = "/proj/irl-agentic/meet/models/landmark_sigir/msmarco_psgs_sigir_gte_base_exps_latent_attn__8_npsg_latent_attn_gran_empty/checkpoint-1000/"
    latent_model_path = os.path.join(model_path,'latent_attn_model')
    config = LatentAttentionConfig.from_pretrained(latent_model_path)
    print(config)
    latent_attention_model = LatentAttentionModel.from_pretrained(latent_model_path,config=config)
    print(latent_attention_model)
    st_model = models.Transformer(latent_model_path,
        config_args={"trust_remote_code": True, "supports_gradient_checkpointing": True},
        model_args={"trust_remote_code": True},
        tokenizer_name_or_path=model_path
    )
    print(st_model)