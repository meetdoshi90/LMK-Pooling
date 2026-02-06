import mteb
from sentence_transformers import models, SentenceTransformer
from transformers import AutoConfig
from transformers.models.modernbert.modeling_modernbert import ModernBertRotaryEmbedding
from typing import override
import torch
from torch import Tensor
import argparse
from sentence_splitter import SentenceSplitter
import json
import os
import logging
from latent_model_utils import *

# Configure basic logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get a logger instance
logger = logging.getLogger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type = str, help="model_name_or_path", default="./dummy_path/")
parser.add_argument("-msl", "--max_seq_len", type = int, help="max_seq_len", default=8192)
parser.add_argument("-pool", "--pool", type = str, help="pooling in landmark,mean,cls,multicls,mean_every_k,latent_attn", default='landmark')
parser.add_argument("-fixed", "--fixed_splitter", type = str2bool, nargs="?", const=True, default=False, help="sentence splitter en or fixed-256")
parser.add_argument("-split_size", "--split_size", type = int, help="split_size for fixed splitter", default=256)
parser.add_argument("-task_type", "--task_type", type = str, help="task_type for eval", default="dummy")
parser.add_argument("-lang", "--lang", type = str, help="lang for eval on multilingual datasets", default="eng")
args, _ = parser.parse_known_args()

def pad_and_stack(tensors, pad_value):
    if not tensors:
        return None
    # print(len(tensors),[x.shape for x in tensors])
    if tensors[0].dim() == 1:
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_value)
    sizes = [t.size(0) for t in tensors]
    maxL = max(sizes)
    padded = [torch.nn.functional.pad(t, (0, maxL - t.size(0), 0, maxL - t.size(0)), value=pad_value) for t in tensors]
    return torch.stack(padded, dim=0)


class LandmarkStage1SentenceTransformer(SentenceTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sentence_splitter = SentenceSplitter(language='en')
        self.query_length = kwargs['tokenizer_kwargs']['query_length'] if 'query_length' in kwargs['tokenizer_kwargs'] else self[0].max_seq_length
        self.document_length = kwargs['tokenizer_kwargs']['document_length'] if 'document_length' in kwargs['tokenizer_kwargs'] else self[0].max_seq_length
        self.normalize = kwargs['config_kwargs']['normalize'] if 'normalize' in kwargs['config_kwargs'] else True

    def cls_pool(self,
                 token_embeddings: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns: tensor of shape (B, D) — the embedding of the first token (CLS token).
        Assumes input_ids[ :, 0 ] corresponds to the CLS token for each sequence.
        token_embeddings: (B, Seq, D)
        attention_mask: (B, Seq)
        """
        # simply take the first token’s embedding for each example:
        return token_embeddings[:, 0, :]

    def mean_pool(self,
                  token_embeddings: torch.Tensor,
                  attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns: tensor of shape (B, D) — mean‐pooled embedding over all non‐padded tokens.
        token_embeddings: (B, Seq, D)
        attention_mask: (B, Seq) with 1 for non-pad tokens, 0 for pad.
        """
        # cast mask to same dtype as embeddings
        mask = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)  # (B, Seq, 1)
        # sum token embeddings
        sum_emb = (token_embeddings * mask).sum(dim=1)  # (B, D)
        # count non-pad tokens (minimum 1 to avoid divide by zero)
        token_count = mask.sum(dim=1).clamp_min(1e-9)  # (B, 1)
        mean_emb = sum_emb / token_count
        return mean_emb
    
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

    @override
    def forward(self, features, task=None):
        '''
        Takes forward pass on the entire sequence, takes mean pool of [LMK] token embeddings, then normalize if necessary
        '''
        # Extract token embeddings from the first module
        output_states = self._first_module().auto_model(
            **{k: features[k] for k in ["input_ids", "attention_mask"]},
            output_hidden_states=True,
            return_dict=True,
        )
        token_embeddings = output_states.last_hidden_state
        # Process token embeddings through subsequent modules if they exist
        if len(list(self._modules.values()))>1:
            for mod in list(self._modules.values())[1:]:
                # Ensure each module receives token embeddings
                features = {'sentence_embedding': token_embeddings, **features}
                features = mod(features)
                # Update token_embeddings after processing
                if args.pool == 'latent_attn':
                    token_embeddings = features['token_embeddings']
                else:
                    token_embeddings = features['sentence_embedding']
        
        if args.pool == 'landmark':
            token_embeddings = self.mean_pool_on_lmk(token_embeddings,features['input_ids'], features['attention_mask'], self[0].tokenizer.sep_token_id)
        elif args.pool == 'mean_every_k': #mean pool at every k tokens without any special LMK token baseline
            token_embeddings = self.mean_pool_on_every_k_tokens(token_embeddings,features['input_ids'], features['attention_mask'], args.split_size)
        elif args.pool == 'mean':
            token_embeddings = self.mean_pool(token_embeddings, features['attention_mask'])
        elif args.pool == 'latent_attn':
            pass # model will return only 1 embedding anyway
        elif args.pool == 'cls':
            token_embeddings = self.cls_pool(token_embeddings, features['attention_mask'])
        elif args.pool == 'multicls':
            token_embeddings = self.mean_pool_on_lmk(token_embeddings,features['input_ids'], features['attention_mask'], self[0].tokenizer.cls_token_id)
        else:
            raise Exception(f'Pooling type not supported: {args.pool}')
        if self.normalize: # normalizer module from sentence transformers normalizes at dim=1 ??!
            token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=-1)
        features.update({'sentence_embedding': token_embeddings})
        return features

    @override
    def tokenize(self, texts: list[str] | list[dict] | list[tuple[str, str]], **kwargs) -> dict[str, Tensor]:
        """
        Tokenizes the texts.

        Args:
            texts (Union[List[str], List[Dict], List[Tuple[str, str]]]): A list of texts to be tokenized.

        Returns:
            Dict[str, Tensor]: A dictionary of tensors with the tokenized texts. Common keys are "input_ids",
                "attention_mask", and "token_type_ids".
        """
        # Move sentence splitter to GPU -> Shouldnt do it here though

        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]
        # Lowercase
        if self[0].do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        # We do fixed length padding since gather operations require same sized tensors and colbert embeddings are a function of seq_len
        # This creates redundancy for query embeddings, simple fix: to limit query max length.
        is_query = kwargs['task']=='query' if 'task' in kwargs else False 
        max_length = self.query_length if is_query else self.document_length
        '''
        Landmark logic for stage 1
        The tokenizer will receive query positives and negatives. 
        Its job is to add [LMK] token at the end of each sentence and return the tokenized sentences with [LMK] tokens.
        For ModernBERT training we will assume [SEP] = [LMK] since by default it is added at the end of each sentence.
        '''
        output_inp_ids = []
        for col in to_tokenize:
            for s in col:
                if s.strip()=="":
                    output_inp_ids.append(torch.tensor([self[0].tokenizer.cls_token_id, self[0].tokenizer.sep_token_id],dtype=torch.long))
                    continue
                assert isinstance(s, str), f'String expected: {type(s)} {s}'
                if args.fixed_splitter or args.pool in ['cls','multicls','mean','mean_every_k','latent_attn']: # if you dont want to split sentences using sentence splitter
                    splitted_text = [s]
                else:
                    try:
                        splitted_text = self.sentence_splitter.split(text=s)
                        # safe use: if after splitting the new text is less than 90% of original text length then keep the original text
                        if len(" ".join(splitted_text)) < (0.9 * len(s)):
                            splitted_text = [s]
                    except Exception as e:
                        logger.info(f'Error {e}') # Timeout catch
                        splitted_text = [s]
                assert len(splitted_text) > 0, f'No sentences to tokenize! {splitted_text} {s}'
                tokenized_item = self[0].tokenizer(
                    splitted_text,
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                    return_attention_mask=False
                )['input_ids']
                if args.fixed_splitter: # chunk into fixed size splits to insert lmk or cls
                    tokenized_item = tokenized_item[0] # since we only keep 1 item in the list for fixed splitter
                    tokenized_item = [tokenized_item[i : i + args.split_size] for i in range(0, len(tokenized_item), args.split_size)]

                if args.pool == 'landmark': # Now we will concatenate the tokenized sentences in [CLS] + S1 + [LMK] + S2 ... [LMK]
                    tokenized_item = [x + [self[0].tokenizer.sep_token_id] for x in tokenized_item]
                    tokenized_item = ([self[0].tokenizer.cls_token_id] + [x for sub in tokenized_item for x in sub])[:max_length]
                elif args.pool == 'multicls': # add [CLS] to every sentence/split
                    tokenized_item = [[self[0].tokenizer.cls_token_id] + x  for x in tokenized_item]
                    tokenized_item = ([x for sub in tokenized_item for x in sub][:max_length-1] + [self[0].tokenizer.sep_token_id])
                else: # cls, mean, latent_attn, mean_every_k
                    tokenized_item = ([self[0].tokenizer.cls_token_id] + [x for sub in tokenized_item for x in sub][:max_length-2] + [self[0].tokenizer.sep_token_id]) # making sure [SEP] token is added at the end always
                output_inp_ids.append(torch.tensor(tokenized_item,dtype=torch.long))
        output_inp_ids = pad_and_stack(output_inp_ids, self[0].tokenizer.pad_token_id)
        attention_mask = torch.where(output_inp_ids!=self[0].tokenizer.pad_token_id, 1, 0).to(torch.bool)
        output['input_ids'] = output_inp_ids
        output['attention_mask'] = attention_mask
        return output
    
    @override
    def save(
        self,
        path: str,
        model_name: str | None = None,
        create_model_card: bool = True,
        train_datasets: list[str] | None = None,
        safe_serialization: bool = True,
    ) -> None:
        """
        Saves a model and its configuration files to a directory, so that it can be loaded
        with ``SentenceTransformer(path)`` again.

        Args:
            path (str): Path on disc where the model will be saved.
            model_name (str, optional): Optional model name.
            create_model_card (bool, optional): If True, create a README.md with basic information about this model.
            train_datasets (list[str], optional): Optional list with the names of the datasets used to train the model.
            safe_serialization (bool, optional): If True, save the model using safetensors. If False, save the model
                the traditional (but unsafe) PyTorch way.
        """
        for module in list(self._modules.values()):
            logger.info(f"Detaching weights from shared memory for module {type(module)}")
            if hasattr(module, "linear") and hasattr(module.linear, "weight"):
                with torch.no_grad():
                    module.linear.weight.data = module.linear.weight.data.clone().contiguous()

        super().save(
            path,
            model_name=model_name,
            create_model_card=create_model_card,
            train_datasets=train_datasets,
            safe_serialization=safe_serialization,
        )

        with open(os.path.join(path, "config_sentence_transformers.json"), "w") as fOut:
            config = self._model_config.copy()
            config["query_length"] = self.query_length
            config["document_length"] = self.document_length
            json.dump(config, fOut, indent=2)
    


model_path = args.model
tokenizer_path = model_path

word_embedding_model = models.Transformer(model_path, 
        max_seq_length=args.max_seq_len,
        config_args={"trust_remote_code": True, "supports_gradient_checkpointing": True},
        model_args={"trust_remote_code": True},
    )

    
# DISABLE_ROPE=True
# if DISABLE_ROPE:
#     def disable_rope(model):
#         with torch.no_grad():
#             for m in model.modules():
#                 if isinstance(m, ModernBertRotaryEmbedding):
#                     m.inv_freq.zero_()
#                     m.attention_scaling = 1.0
#     disable_rope(word_embedding_model.auto_model)


modules = [word_embedding_model]

if args.pool == 'latent_attn':
    latent_attn_model_path = os.path.join(model_path, 'latent_attn_model')
    if not os.path.exists(latent_attn_model_path):
        raise NotADirectoryError(f'Could not find latent attn model parameters in the given directory: {latent_attn_model_path}')
    
    LATENT_ATTENTION_TYPE = "latent_attention"
    AutoConfig.register(LATENT_ATTENTION_TYPE, LatentAttentionConfig)
    AutoModel.register(LatentAttentionConfig, LatentAttentionModel)
    LatentAttentionModel.register_for_auto_class()

    latent_attention_model = models.Transformer(latent_attn_model_path,
        config_args={"trust_remote_code": True, "supports_gradient_checkpointing": True},
        model_args={"trust_remote_code": True},
        tokenizer_name_or_path=model_path
    )
    modules.append(latent_attention_model)
    
model = LandmarkStage1SentenceTransformer(
    modules=modules, 
    trust_remote_code=True,
    model_kwargs={'torch_dtype': torch.bfloat16},
    tokenizer_kwargs = {
        "padding_side": "right",
        "truncation_side": "right",
        "model_max_length": args.max_seq_len,
        "query_length": args.max_seq_len,
        "document_length": args.max_seq_len
        },
    config_kwargs = {
        'normalize': True
    }
)

model_path = "_".join(model_path.split('/')[-3:])
if args.task_type == "dummy":
    TASK_NAMES=['ArguAna', 'FiQA2018']
elif args.task_type == "mteb_eng_benchmark":
    TASK_NAMES=['MTEB(eng, v2)']
elif args.task_type == 'eurlex_multilingual':
    TASK_NAMES=['MultiEURLEXMultilabelClassification']
elif args.task_type == "mteb_v2":
    TASK_NAMES=['ArguAna','FiQA2018','SCIDOCS',"CQADupstackUnixRetrieval","CQADupstackGamingRetrieval","ClimateFEVERHardNegatives",'FEVERHardNegatives','HotpotQAHardNegatives',"TRECCOVID","Touche2020Retrieval.v3"]
elif args.task_type == "beir-15":
    TASK_NAMES=['NFCorpus','NQ','HotpotQA',"Touche2020","CQADupstackRetrieval","QuoraRetrieval",'DBPedia',"FEVER","ClimateFEVER","SciFact"] #add these if not already using MTEBv2 and MSMARCO ['TRECCOVID','FiQA2018','ArguAna','SCIDOCS','MSMARCO']
elif args.task_type == "mldr":
    TASK_NAMES=['MultiLongDocRetrieval']
elif args.task_type == "miracl_hn":
    TASK_NAMES=['MIRACLRetrievalHardNegatives']
elif args.task_type == "msmarco":
    TASK_NAMES=['MSMARCO']
elif args.task_type == "long_embed":
    TASK_NAMES=['LEMBNeedleRetrieval', 'LEMBPasskeyRetrieval','LEMBQMSumRetrieval','LEMBSummScreenFDRetrieval','LEMBWikimQARetrieval','LEMBNarrativeQARetrieval'] 
elif args.task_type == "coir":
    TASK_NAMES=["AppsRetrieval",
                "CodeFeedbackMT",
                "CodeFeedbackST",
                "CodeTransOceanContest",
                "CodeTransOceanDL",
                "CosQA",
                "SyntheticText2SQL",
                "StackOverflowQA",
                "COIRCodeSearchNetRetrieval",
                "CodeSearchNetCCRetrieval"]
else:
    TASK_NAMES=[args.task_type]
    # raise Exception(f'Task types not defined')

MULTILINGUAL_TASK_NAMES = ['MultiLongDocRetrieval','MIRACLRetrievalHardNegatives','MultiEURLEXMultilabelClassification']

if TASK_NAMES==['MTEB(eng, v2)']:
    tasks = mteb.get_benchmark("MTEB(eng, v2)")
    print(tasks)
    output_path = f'mteb_results/{args.lang}_mtebengbenchmark_{args.pool}_{str(args.fixed_splitter)}_{args.split_size}_{args.max_seq_len}_{model_path.replace('/',"_")}'
    with torch.no_grad():
        out = mteb.evaluate(
                    model=model,
                    tasks=tasks,
                    overwrite_strategy='always',
                    cache=None,
                    encode_kwargs={'batch_size':16}
                )
        print(out)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for tr in out.task_results:
        if hasattr(tr, "scores") and tr.scores is not None:
            with open(f"{output_path}/{tr.task_name}.json", "w") as f:
                json.dump(tr.scores, f, indent=2)
    if 'test' in out[0].scores:
        score = out[0].scores['test'][0]['main_score']
    elif 'dev' in out[0].scores:
        score = out[0].scores['dev'][0]['main_score']
    elif 'train' in out[0].scores:
        score = out[0].scores['train'][0]['main_score']
    else:
        score = out
    
    print(TASK_NAMES, score)
    exit()

for TASK_NAME in TASK_NAMES:
    eval_splits = ['dev'] if ('msmarco' in TASK_NAME.lower() or 'miracl' in TASK_NAME.lower()) else ['test']
    eval_splits = eval_splits if 'lemb' not in TASK_NAME.lower() else None
    languages = [args.lang] if (TASK_NAME in MULTILINGUAL_TASK_NAMES) else None
    if languages!=None:
        task = mteb.get_task(TASK_NAME, languages=languages, eval_splits=eval_splits)
        print(task.languages)
    else:
        task = mteb.get_task(TASK_NAME)
    output_path = f'results_new/{args.lang}_{TASK_NAME}_{args.pool}_{str(args.fixed_splitter)}_{args.split_size}_{args.max_seq_len}_{model_path.replace('/',"_")}'
    with torch.no_grad():
        if TASK_NAME in MULTILINGUAL_TASK_NAMES:
            out = mteb.evaluate(
                            model=model,
                            tasks=task,
                            overwrite_strategy='always',
                            cache=None,
                            # prediction_folder=f'predictions_new/{TASK_NAME}_{args.pool}_{str(args.fixed_splitter)}_{args.split_size}_{args.max_seq_len}_{model_path.replace('/',"_")}',
                            encode_kwargs={'batch_size':16}
                        )
        else:
            out = mteb.evaluate(
                            model=model,
                            tasks=[task],
                            overwrite_strategy='always',
                            cache=None,
                            # prediction_folder=f'predictions_new/{TASK_NAME}_{args.pool}_{str(args.fixed_splitter)}_{args.split_size}_{args.max_seq_len}_{model_path.replace('/',"_")}',
                            encode_kwargs={'batch_size':32 if 'lemb' not in TASK_NAME.lower() else 2}
                        )
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(f"{output_path}/{TASK_NAME}.json", "w") as f:
        if len(out.task_results)>1:
            all_scores = [
                tr.scores for tr in out.task_results
                if hasattr(tr, "scores") and tr.scores is not None
            ]
            json.dump(all_scores, f, indent=2)
        else:
            print(out.task_results[0].scores)
            json.dump(out.task_results[0].scores, f, indent=2)
    if 'test' in out[0].scores:
        score = out[0].scores['test'][0]['main_score']
    elif 'dev' in out[0].scores:
        score = out[0].scores['dev'][0]['main_score']
    elif 'train' in out[0].scores:
        score = out[0].scores['train'][0]['main_score']
    else:
        score = out
    
    print(TASK_NAME, score)