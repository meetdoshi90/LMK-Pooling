import logging
import os

import torch
from torch import nn
from transformers import BertForMaskedLM, AutoModelForMaskedLM, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput

from .arguments import ModelArguments
from .enhancedDecoder import BertLayerForDecoder

logger = logging.getLogger(__name__)


class RetroMAEForPretraining(nn.Module):
    def __init__(
            self,
            model: AutoModelForMaskedLM,
            model_args: ModelArguments,
    ):
        super(RetroMAEForPretraining, self).__init__()
        self.lm = model

        if hasattr(self.lm, 'bert'):
            self.decoder_embeddings = self.lm.bert.embeddings
        elif hasattr(self.lm, 'roberta'):
            self.decoder_embeddings = self.lm.roberta.embeddings
        elif hasattr(self.lm, 'new'):
            self.decoder_embeddings = self.lm.new.embeddings
        else:
            # generic "model" attribute used in ModernBERT
            self.decoder_embeddings = self.lm.model.embeddings
            # change to rotary positional embeddings for the decoder
            model.config.position_embedding_type = "relative_key_query"

        self.c_head = BertLayerForDecoder(model.config)
        self.c_head.apply(self.lm._init_weights)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.model_args = model_args
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.lm.gradient_checkpointing_enable(**kwargs)

    def forward(self,
                encoder_input_ids, encoder_attention_mask, encoder_labels,
                decoder_input_ids, decoder_attention_mask, decoder_labels):

        lm_out: MaskedLMOutput = self.lm(
            encoder_input_ids, encoder_attention_mask,
            labels=encoder_labels,
            output_hidden_states=True,
            return_dict=True
        )
        if self.lm.config.model_type == "modernbert":
            # modernbert outputs a 2D vector, not 3D-- may need to repad
            padded_hidden_state = self.pad_modernbert_hidden_states(encoder_input_ids, encoder_attention_mask, [lm_out.hidden_states[-1]], batch_size=lm_out.logits.shape[0], seq_len=lm_out.logits.shape[1])
            cls_hiddens = padded_hidden_state[-1][:, :1] # B 1 D
        else:
            cls_hiddens = lm_out.hidden_states[-1][:, :1]  # B 1 D

        if self.lm.config.model_type == "new":
            # for gte-en-mlm-base models
            decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids, unpad_inputs = False) #  return embeddings, attention_mask, rope_embeds, length
            decoder_embedding_output = decoder_embedding_output[0]
        else:
            decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)
        
        hiddens = torch.cat([cls_hiddens, decoder_embedding_output[:, 1:]], dim=1)

        # if hasattr(self.lm, 'roberta'):
        #     decoder_position_ids = self.lm.roberta.embeddings.position_ids[:, :decoder_input_ids.size(1)]
        #     decoder_position_embeddings = self.lm.roberta.embeddings.position_embeddings(decoder_position_ids)  # B L D
        #     query = decoder_position_embeddings + cls_hiddens
        # else:
        #     decoder_position_ids = self.lm.bert.embeddings.position_ids[:, :decoder_input_ids.size(1)]
        #     decoder_position_embeddings = self.lm.bert.embeddings.position_embeddings(decoder_position_ids)  # B L D
        #     query = decoder_position_embeddings + cls_hiddens

        cls_hiddens = cls_hiddens.expand(hiddens.size(0), hiddens.size(1), hiddens.size(2))
        if self.lm.config.model_type == "new":
            # for gte-en-mlm-base models
            query = self.decoder_embeddings(inputs_embeds=cls_hiddens, unpad_inputs = False) #  return embeddings, attention_mask, rope_embeds, length
            query = query[0]
        else:
            query = self.decoder_embeddings(inputs_embeds=cls_hiddens)

        matrix_attention_mask = self.lm.get_extended_attention_mask(
            decoder_attention_mask,
            decoder_attention_mask.shape,
            decoder_attention_mask.device
        )

        hiddens = self.c_head(query=query,
                              key=hiddens,
                              value=hiddens,
                              attention_mask=matrix_attention_mask)[0]
        pred_scores, loss = self.mlm_loss(hiddens, decoder_labels)

        return (loss + lm_out.loss,)

    def mlm_loss(self, hiddens, labels):
        if hasattr(self.lm, 'cls'):
            pred_scores = self.lm.cls(hiddens)
        elif hasattr(self.lm, 'lm_head'):
            pred_scores = self.lm.lm_head(hiddens)
        elif self.lm.config.model_type == "modernbert":
            # for modernbert
            # check if the decoder and head is merged into self.lm.compiled_head -> merged dynamically acc to torch version
            pred_scores = (
                self.lm.compiled_head(hiddens)
                if self.lm.config.reference_compile
                else self.lm.decoder(self.lm.head(hiddens))
            )
        else:
            raise NotImplementedError

        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return pred_scores, masked_lm_loss

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(os.path.join(output_dir, "encoder_model"))
        torch.save(self.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args)
        return model

    def pad_modernbert_hidden_states(self, input_ids, attention_mask, all_hidden_states, batch_size, seq_len):
        from transformers.models.modernbert.modeling_modernbert import  _pad_modernbert_output
        with torch.no_grad():
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

            all_hidden_states_padded = []
            for hs in all_hidden_states:
                all_hidden_states_padded.append(_pad_modernbert_output(inputs=hs, indices=indices, batch=batch_size, seqlen=seq_len))

        return all_hidden_states_padded

class LandmarkRetroMAEForPretraining(RetroMAEForPretraining):
    """
    Uses landmark tokens instead of CLS representation for retromae training
    Landmark embeddings: https://aclanthology.org/2024.acl-long.180/
    """
    def __init__(
            self,
            model: AutoModelForMaskedLM,
            model_args: ModelArguments,
            tokenizer: AutoTokenizer
    ):
        super().__init__(model, model_args)
        self.tokenizer = tokenizer
        if self.tokenizer.sep_token_id:
            self.landmark_token_id = self.tokenizer.sep_token_id
        else:
            self.landmark_token_id = self.tokenizer.eos_token_id
        

    def forward(self,
                encoder_input_ids, encoder_attention_mask, encoder_labels,
                decoder_input_ids, decoder_attention_mask, decoder_labels):

        lm_out: MaskedLMOutput = self.lm(
            encoder_input_ids, encoder_attention_mask,
            labels=encoder_labels,
            output_hidden_states=True,
            return_dict=True
        )

        if self.lm.config.model_type == "modernbert":
            # modernbert outputs a 2D vector, not 3D-- may need to repad
            padded_hidden_state = self.pad_modernbert_hidden_states(encoder_input_ids, encoder_attention_mask, [lm_out.hidden_states[-1]], batch_size=lm_out.logits.shape[0], seq_len=lm_out.logits.shape[1])
        else:
            padded_hidden_state = lm_out.hidden_states
        
        # cls_hiddens = padded_hidden_state[-1][:, :1] # B 1 D
        lmk_hiddens = self.mean_pool_on_lmk(padded_hidden_state[-1],encoder_input_ids, encoder_attention_mask, self.tokenizer.landmark_token_id) # B D
        lmk_hiddens = lmk_hiddens.unsqueeze(1) # B 1 D

        if self.lm.config.model_type == "new":
            # for gte-en-mlm-base models
            decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids, unpad_inputs = False) #  return embeddings, attention_mask, rope_embeds, length
            decoder_embedding_output = decoder_embedding_output[0]
        else:
            decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)
        hiddens = torch.cat([lmk_hiddens, decoder_embedding_output[:, 1:]], dim=1)

        lmk_hiddens = lmk_hiddens.expand(hiddens.size(0), hiddens.size(1), hiddens.size(2))
        if self.lm.config.model_type == "new":
            # for gte-en-mlm-base models
            query = self.decoder_embeddings(inputs_embeds=lmk_hiddens, unpad_inputs = False) #  return embeddings, attention_mask, rope_embeds, length
            query = query[0]
        else:
            query = self.decoder_embeddings(inputs_embeds=lmk_hiddens)
    

        matrix_attention_mask = self.lm.get_extended_attention_mask(
            decoder_attention_mask,
            decoder_attention_mask.shape,
            decoder_attention_mask.device
        )

        hiddens = self.c_head(query=query,
                              key=hiddens,
                              value=hiddens,
                              attention_mask=matrix_attention_mask)[0]
        pred_scores, loss = self.mlm_loss(hiddens, decoder_labels)

        return (loss + lm_out.loss,)
    
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

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        tokenzier = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        model = cls(hf_model, model_args, tokenzier)
        return model