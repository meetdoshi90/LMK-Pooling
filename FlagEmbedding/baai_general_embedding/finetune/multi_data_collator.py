import torch

from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import DataCollatorWithPadding, BatchEncoding
import os
import random
from .arguments import DataArguments, TrainingArguments
import math 

def _unpack_doc_values(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    doc_examples = []
    for f in features:
        keys = list(f.keys())
        lists_per_key = len(f[keys[0]])
        for idx in range(lists_per_key):
            doc_examples.append({k: f[k][idx] for k in keys})
    return doc_examples


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


@dataclass
class DistilEmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    args: DataArguments = None
    train_args: TrainingArguments = None
    sentence_splitter: Any = None

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score
    
    def lmk_tokenizer(self, texts: List[str], max_len: int = 100000):
        tokenized_inputs = []
        output = {}
        for text in texts:
            if self.train_args.sentence_pooling_method == 'lmk_en':
                splitted_text = self.sentence_splitter.split(text)
            elif self.train_args.sentence_pooling_method == 'lmk_multi':
                splitted_text = self.sentence_splitter.split_longtext(text=text, lang='xx') #TODO: pass lang
                if splitted_text==[]:
                    splitted_text = [text]
            elif self.train_args.sentence_pooling_method == 'lmk_fixed' or self.train_args.sentence_pooling_method == 'lmk_var':
                splitted_text = [text]
            else:
                raise NotImplementedError(self.train_args.sentence_pooling_method)

            tokenized_item = self.tokenizer(
                splitted_text,
                padding=False,
                truncation=False,
                add_special_tokens=False,
                return_attention_mask=False
            )['input_ids']

            if self.train_args.sentence_pooling_method == 'lmk_fixed':
                tokenized_item = tokenized_item[0] # since we only added 1 item to the list
                tokenized_item = [tokenized_item[i : i + int(self.train_args.landmark_granularity_val)] for i in range(0, len(tokenized_item), int(self.train_args.landmark_granularity_val))]
            elif self.train_args.sentence_pooling_method == 'lmk_var':
                available_granularities = self.train_args.landmark_granularity_val.split('_')
                available_granularities = [int(x) for x in available_granularities]
                tokenized_item = tokenized_item[0]
                selected_granularity = random.choice(available_granularities) # choose one granularity at random
                tokenized_item = [tokenized_item[i : i + selected_granularity] for i in range(0, len(tokenized_item), selected_granularity)]
            
            # Add [CLS] [SEP] and clip to max len
            tokenized_item = [x + [self.tokenizer.sep_token_id] for x in tokenized_item] # add sep after each sentence
            tokenized_item = [self.tokenizer.cls_token_id] + [x for sub in tokenized_item for x in sub] # add cls token
            tokenized_item = tokenized_item[:max_len] # clip to max_len
            if self.tokenizer.sep_token_id not in tokenized_item:
                tokenized_item[-1] = self.tokenizer.sep_token_id
            tokenized_inputs.append(torch.tensor(tokenized_item,dtype=torch.long))
        tokenized_inputs = pad_and_stack(tokenized_inputs, self.tokenizer.pad_token_id)
        attention_mask = torch.where(tokenized_inputs!=self.tokenizer.pad_token_id, 1, 0).to(torch.bool)
        output['input_ids'] = tokenized_inputs
        output['attention_mask'] = attention_mask
        return output

    def __call__(self, features):
        try:
            query = [f['query'] for f in features]
            f = features[0]
            neg_key = "neg" if "neg" in f else "negatives"
            pos_key = "pos" if "pos" in f else "positives"
            passage_list = []
            for f in features:
                if pos_key=="pos":
                    if 'pos_scores' in f and f['pos_scores']!=None:
                        passage_list.append((f[pos_key][0], f['pos_scores'][0]) if isinstance(f[pos_key], list) else f[pos_key])
                    else:
                        passage_list.append((f[pos_key][0], 100.0) if isinstance(f[pos_key], list) else f[pos_key])
                else:
                    passage_list.append((f[pos_key]["docs"][0], f[pos_key]['score'][0]) if isinstance(f[pos_key]["docs"], list) else f[pos_key]["docs"])
                if neg_key in f:
                    if neg_key=="neg":
                        if f[neg_key]!=None:
                            negs = [x for x in f[neg_key]]
                            if 'neg_scores' in f and f['neg_scores']!=None:
                                neg_scores = f['neg_scores']
                            else:
                                neg_scores = [-100.0]*len(negs)
                        else:
                            negs = []
                        negs = [(x,y) for x,y in zip(negs,neg_scores)]
                    else:
                        negs = f[neg_key]["docs"]
                        neg_scores = f[neg_key]["score"]
                        negs = [(x,y) for x,y in zip(negs,neg_scores)]
                else:
                    negs = []
                if negs!=[]:
                    negs = random.choices(negs, k=self.args.train_group_size-1)
                    passage_list.extend(negs)
            passage = [x[0] for x in passage_list]
            score_list = [x[1] for x in passage_list]
            if 'weight' in features[0]:
                weights = [f['weight'] for f in features]
            else:
                weights = []
        except:
            query = [f[0] for f in features]
            passage = [f[1] for f in features]
            weights = [1.0 for f in features]
            

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])
        if 'lmk' not in self.train_args.sentence_pooling_method:
            q_collated = self.tokenizer(
                query,
                padding="longest",
                truncation=True,
                max_length=self.query_max_len,
                return_tensors="pt",
            )
            d_collated = self.tokenizer(
                passage,
                padding="longest",
                truncation=True,
                max_length=self.passage_max_len,
                return_tensors="pt",
            )
        else:
            q_collated = self.lmk_tokenizer(query, max_len=self.query_max_len)
            d_collated = self.lmk_tokenizer(passage, max_len=self.passage_max_len)
        score_list = [s if s is not None else 0.0 for s in score_list]
        weights = [w if w is not None else 1.0 for w in weights]
        return {"query": q_collated, "passage": d_collated, "teacher_score": torch.tensor(score_list)}
    
    
@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    args: DataArguments = None
    train_args: TrainingArguments = None
    sentence_splitter: Any = None

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score
    
    def lmk_tokenizer(self, texts: List[str], max_len: int = 100000):
        tokenized_inputs = []
        output = {}
        for text in texts:
            if self.train_args.sentence_pooling_method == 'lmk_en':
                splitted_text = self.sentence_splitter.split(text)
            elif self.train_args.sentence_pooling_method == 'lmk_multi':
                splitted_text = self.sentence_splitter.split_longtext(text=text, lang='xx') #TODO: pass lang
                if splitted_text==[]:
                    splitted_text = [text]
            elif self.train_args.sentence_pooling_method == 'lmk_fixed' or self.train_args.sentence_pooling_method == 'lmk_var':
                splitted_text = [text]
            else:
                raise NotImplementedError(self.train_args.sentence_pooling_method)

            tokenized_item = self.tokenizer(
                splitted_text,
                padding=False,
                truncation=False,
                add_special_tokens=False,
                return_attention_mask=False
            )['input_ids']

            if self.train_args.sentence_pooling_method == 'lmk_fixed':
                tokenized_item = tokenized_item[0] # since we only added 1 item to the list
                tokenized_item = [tokenized_item[i : i + int(self.train_args.landmark_granularity_val)] for i in range(0, len(tokenized_item), int(self.train_args.landmark_granularity_val))]
            elif self.train_args.sentence_pooling_method == 'lmk_var':
                available_granularities = self.train_args.landmark_granularity_val.split('_')
                available_granularities = [int(x) for x in available_granularities]
                tokenized_item = tokenized_item[0]
                selected_granularity = random.choice(available_granularities) # choose one granularity at random
                tokenized_item = [tokenized_item[i : i + selected_granularity] for i in range(0, len(tokenized_item), selected_granularity)]
            
            # Add [CLS] [SEP] and clip to max len
            tokenized_item = [x + [self.tokenizer.sep_token_id] for x in tokenized_item] # add sep after each sentence
            tokenized_item = [self.tokenizer.cls_token_id] + [x for sub in tokenized_item for x in sub] # add cls token
            tokenized_item = tokenized_item[:max_len] # clip to max_len
            if self.tokenizer.sep_token_id not in tokenized_item:
                tokenized_item[-1] = self.tokenizer.sep_token_id
            tokenized_inputs.append(torch.tensor(tokenized_item,dtype=torch.long))
        tokenized_inputs = pad_and_stack(tokenized_inputs, self.tokenizer.pad_token_id)
        attention_mask = torch.where(tokenized_inputs!=self.tokenizer.pad_token_id, 1, 0).to(torch.bool)
        output['input_ids'] = tokenized_inputs
        output['attention_mask'] = attention_mask
        return output

    def __call__(self, features):
        try:
            query = []
            # f = features[0]
            passage_list = []
            for f in features:
                if f is None:
                    continue
                neg_key = "neg" if "neg" in f else ("negatives" if "negatives" in f else None)
                pos_key = "pos" if "pos" in f else ("positives" if "positives" in f else None)
                if pos_key=="pos":
                    passage_list.append(f[pos_key][0] if isinstance(f[pos_key], list) else f[pos_key])
                    query.append(f['query'])
                elif pos_key=='positives' and f[pos_key]!=None:
                    passage_list.append(f[pos_key]["docs"][0] if isinstance(f[pos_key]["docs"], list) else f[pos_key]["docs"])
                    query.append(f['query'])
                else:
                    continue
                if neg_key in f:
                    if neg_key=="neg":
                        negs = f[neg_key]
                    elif neg_key=="negatives" and f[neg_key]!=None:
                        negs = f[neg_key]["docs"]
                    else:
                        negs = []
                else:
                    negs = []
                if negs!=[]:
                    negs = random.choices(negs, k=self.args.train_group_size-1)
                    passage_list.extend(negs) 
            passage = passage_list
            if 'weight' in features[0]:
                weights = [f['weight'] for f in features]
            else:
                weights = []
        except Exception as e:
            print(f'Exception occured {e} {features}')
            query = [f[0] for f in features]
            passage = [f[1] for f in features]
            weights = [1.0 for f in features]
            

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])
        for i in range(len(passage)):
            if passage[i]==None:
                passage[i] = ""
        weights = [w if w is not None else 1.0 for w in weights]
        if 'lmk' not in self.train_args.sentence_pooling_method:
            q_collated = self.tokenizer(
                query,
                padding="longest",
                truncation=True,
                max_length=self.query_max_len,
                return_tensors="pt",
            )
            d_collated = self.tokenizer(
                passage,
                padding="longest",
                truncation=True,
                max_length=self.passage_max_len,
                return_tensors="pt",
            )
        else:
            q_collated = self.lmk_tokenizer(query, max_len=self.query_max_len)
            d_collated = self.lmk_tokenizer(passage, max_len=self.passage_max_len)
        return {"query": q_collated, "passage": d_collated}
    
@dataclass
class InBatchEmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        try:
            query = [f['query'] for f in features]
            #passage = [f['pos'][0] for f in features]
            f = features[0]
            pos_key = "pos" if "pos" in f else "positives"
            passage_list = []
            for f in features:
                if pos_key=="pos":
                    passage_list.append(f[pos_key][0] if isinstance(f[pos_key], list) else f[pos_key])
                else:
                    passage_list.append(f[pos_key]["docs"][0] if isinstance(f[pos_key]["docs"], list) else f[pos_key]["docs"])
            passage = passage_list
        except:
            query = [f[0] for f in features]
            passage = [f[1] for f in features]
            

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])
        q_collated = self.tokenizer(
            query,
            padding="longest",
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding="longest",
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated}