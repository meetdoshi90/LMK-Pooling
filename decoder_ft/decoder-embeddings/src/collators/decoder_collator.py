import torch

from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import DataCollatorWithPadding, BatchEncoding
from collators.collator_utils import _get_examples
import torch.distributed as dist
from loaders.loader_utils import group_doc_ids, dataset_attributes
from collections import deque
import random
from transformers.file_utils import PaddingStrategy
from loaders.loader_utils import group_doc_ids, dataset_attributes, insert_lmk_tokens, sample_lmk_granularity
import os 
import gc


@dataclass
class DecoderCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, reranker_tokenizer, pad_to_multiple_of, args):
        super(DecoderCollator, self).__init__(tokenizer, pad_to_multiple_of=pad_to_multiple_of)
        self.tokenizer = tokenizer
        self.reranker_tokenizer = reranker_tokenizer
        self.args = args
        # ── LMK setup ────────────────────────────────────────────────────────
        self._use_lmk       = (getattr(args, 'pooling_source', '') == 'lmk')
        self._lmk_gran      = getattr(args, 'lmk_granularity', 64)
        raw_set             = getattr(args, 'lmk_granularity_set', None)
        self._lmk_gran_set  = (
            [int(g.strip()) for g in raw_set.split(',') if g.strip()]
            if raw_set else None
        )

    def _tokenize_with_lmk_or_eos(
        self,
        texts: List[str],
        max_length: int,
    ) -> List[List[int]]:
        """
        For LMK pooling: tokenize then insert EOS as landmarks every N tokens.
        For mean/last:   tokenize with max_length-1 and append EOS once (original behaviour).
        """
        eos_id = self.tokenizer.eos_token_id
        if self._use_lmk:
            # Pre-truncate to leave room for inserted EOS landmarks
            budget = max_length - (max_length // self._lmk_gran) - 1
            raw = self.tokenizer(
                texts,
                max_length=budget,
                return_attention_mask=False,
                padding=PaddingStrategy.DO_NOT_PAD,
                add_special_tokens=True,
                truncation=True,
            )
            result = []
            for ids in raw['input_ids']:
                gran = sample_lmk_granularity(self._lmk_gran, self._lmk_gran_set)
                result.append(insert_lmk_tokens(ids, gran, eos_id, max_length))
            return result
        else:
            raw = self.tokenizer(
                texts,
                max_length=max_length - 1,
                return_attention_mask=False,
                padding=PaddingStrategy.DO_NOT_PAD,
                truncation=True,
            )
            return [ids + [eos_id] for ids in raw['input_ids']]


    def preprocess_data(self, examples):
        # rank = dist.get_rank()

        if '_instruction' in examples[0]:
            instruction  = examples[0]['_instruction']
            max_length_q = examples[0]['_max_length_q']
            max_length_p = examples[0]['_max_length_p']
        else:
            # Fallback for data that predates the metadata stamping
            instruction  = 'Given a web search query, retrieve relevant passages that answer the query'
            max_length_q = self.args.q_max_len
            max_length_p = self.args.p_max_len
            
        if 'pos' in examples[0]: #bge format data
            query = [x['query'] for x in examples]
            positives = [x['pos'] for x in examples]
            negatives = [x['neg'] for x in examples]
            positive_scores = [x['pos_scores'] for x in examples]
            negative_scores = [x['neg_scores'] for x in examples]
            new_examples = {
                'query_id': [],
                'query': [],
                'positives': [],
                'negatives': []
            }
            idx = 0
            for i in range(len(query)):
                new_examples['query_id'].append(str(idx))
                new_examples['query'].append(query[i])
                idx+=1
                if positive_scores[i]==None:
                    positive_scores[i] = [100.0]*len(positives[i])
                if negative_scores[i]==None:
                    negative_scores[i] = [100.0]*len(negatives[i])
                new_examples['positives'].append({
                    'doc_id': [str(idx+j) for j in range(len(positive_scores[i]))],
                    'score': positive_scores[i],
                    'docs': positives[i],
                    'titles': ['']*len(positive_scores[i])
                })
                idx+=len(positive_scores[i])
                new_examples['negatives'].append({
                    'doc_id': [str(idx+j) for j in range(len(negative_scores[i]))],
                    'score': negative_scores[i],
                    'docs': negatives[i],
                    'titles': ['']*len(negative_scores[i])
                })
                idx+=len(negative_scores[i])
            examples = new_examples
        else:
            new_examples = {
                'query_id': [],
                'query': [],
                'positives': [],
                'negatives': []
            }
            for ex in examples:
                new_examples['query_id'].append(ex['query_id'])
                new_examples['query'].append(ex['query'])
                new_examples['positives'].append(ex['positives'])
                new_examples['negatives'].append(ex['negatives'])
            examples = new_examples

        # print(examples.keys())
        if self.args.train_n_passages > 1:
            if 'doc_id' not in examples['negatives'][0] or len(examples['negatives'][0]['doc_id']) == 0:
                rotate_positive_as_negative = True
                assert self.args.train_n_passages == 2, "randomly create hard negative only create singel negative for now"
                temp_positives = examples['positives']
                temp_positives = deque(temp_positives)
                temp_positives.rotate(1)
                temp_positives=list(temp_positives)
                examples['negatives'] = temp_positives

        input_doc_ids: List[int] = group_doc_ids(
            examples=examples,
            negative_size=self.args.train_n_passages-1,
            offset=random.randint(0,60)+ self.args.seed,
            use_first_positive=self.args.use_first_positive
        )
        assert len(input_doc_ids) == len(examples['query']) * self.args.train_n_passages

        #input_docs: List[str] = [self.corpus[doc_id]['contents'] for doc_id in input_doc_ids]
        #input_titles: List[str] = [self.corpus[doc_id]['title'] for doc_id in input_doc_ids]
        all_doc_ids = []
        all_doc_texts = []
        all_doc_titles = []
        for i in range(len(examples['query'])):
            all_doc_ids.extend(examples['positives'][i]['doc_id']) 
            all_doc_texts.extend(examples['positives'][i]['docs'])
            all_doc_titles.extend(examples['positives'][i]['titles'])
            if (self.args.train_n_passages-1):
                all_doc_ids.extend(examples['negatives'][i]['doc_id'])
                all_doc_texts.extend(examples['negatives'][i]['docs'])
                all_doc_titles.extend(examples['negatives'][i]['titles'])
        # all_doc_ids = [ int(d) for d in all_doc_ids ]
        doc_text = dict(zip(all_doc_ids,all_doc_texts))
        doc_titles = dict(zip(all_doc_ids,all_doc_titles))
        
        input_docs: List[str] = [doc_text[doc_id] for doc_id in input_doc_ids]
        # input_titles: List[str] = [doc_titles[doc_id] for doc_id in input_doc_ids]

        if self.reranker_tokenizer != None:
            reranker_batch_text = []
            for q in examples['query']:
                for d in input_docs:
                    reranker_batch_text.append([q,d])
            reranker_batch_dict = self.reranker_tokenizer(
                reranker_batch_text,
                max_length=self.reranker_tokenizer.model_max_length,
                padding=PaddingStrategy.DO_NOT_PAD,
                return_attention_mask=False,
                truncation=True
            )
        else:
            reranker_batch_dict = None


        def _get_detailed_instruct(task_description: str, query: str) -> str:
            return f'Instruct: {task_description}\nQuery: {query}'

        for i, q in enumerate(examples['query']):
            examples['query'][i] = _get_detailed_instruct(instruction, q)

        # ── CHANGED: use unified tokenise helper instead of inline EOS append ─
        q_input_ids = self._tokenize_with_lmk_or_eos(examples['query'],    max_length_q)
        d_input_ids = self._tokenize_with_lmk_or_eos(input_docs,           max_length_p)

        query_batch_dict = {'input_ids': q_input_ids}
        doc_batch_dict   = {'input_ids': d_input_ids}

        def _merge_dict(query_batch_dict, doc_batch_dict, reranker_batch_dict=None):
            merged_dict = {'q_{}'.format(k): v for k, v in query_batch_dict.items()}
            step_size = self.args.train_n_passages
            for k, v in doc_batch_dict.items():
                k = 'd_{}'.format(k)
                merged_dict[k] = []
                for idx in range(0, len(v), step_size):
                    merged_dict[k].append(v[idx:(idx + step_size)])
            
            if reranker_batch_dict != None:
                for k, v in reranker_batch_dict.items():
                    k = f'r_{k}'
                    num_queries = len(merged_dict['q_input_ids'])
                    merged_dict[k] = []
                    for idx in range(0, len(v), num_queries*step_size):
                        merged_dict[k].append(v[idx:(idx + (num_queries*step_size))])

            if self.args.do_kd_biencoder:
                qid_to_doc_id_to_score = {}

                def _update_qid_pid_score(q_id: str, ex: Dict):
                    # assert len(ex['doc_id']) == len(ex['score']), '{} != {}'.format(len(ex['doc_id']), len(ex['score']))
                    if len(ex['doc_id']) != len(ex['score']):
                        print('{} != {}'.format(len(ex['doc_id']), len(ex['score'])))
                        ex['score'] = [0.0]*len(ex['doc_id'])
                    if q_id not in qid_to_doc_id_to_score:
                        qid_to_doc_id_to_score[q_id] = {}
                    for doc_id, score in zip(ex['doc_id'], ex['score']):
                        qid_to_doc_id_to_score[q_id][doc_id] = score

                for idx, query_id in enumerate(examples['query_id']):
                    _update_qid_pid_score(query_id, examples['positives'][idx])
                    _update_qid_pid_score(query_id, examples['negatives'][idx])

                merged_dict['kd_labels'] = []
                for idx in range(0, len(input_doc_ids), step_size):
                    qid = examples['query_id'][idx // step_size]
                    cur_kd_labels = [qid_to_doc_id_to_score[qid][doc_id] for doc_id in input_doc_ids[idx:idx + step_size]]
                    merged_dict['kd_labels'].append(cur_kd_labels)
                assert len(merged_dict['kd_labels']) == len(examples['query_id']), \
                    '{} != {}'.format(len(merged_dict['kd_labels']), len(examples['query_id']))
            return merged_dict
        
        merged_dict = _merge_dict(query_batch_dict, doc_batch_dict, reranker_batch_dict)

        new_merged_list = []
        for i in range(len(merged_dict['q_input_ids'])):
            new_merged_list.append(
                {
                    k: merged_dict[k][i] for k in merged_dict.keys()
                }
            )
        # gc.collect()
        # Custom formatting function must return a dict
        return new_merged_list

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:

        features = self.preprocess_data(features)
        q_prefix, d_prefix, r_prefix = 'q_', 'd_', 'r_'
        query_examples, doc_examples, reranker_examples = _get_examples(q_prefix, d_prefix, r_prefix, features)
        q_collated = self.tokenizer.pad(
            query_examples,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors=self.return_tensors)
        d_collated = self.tokenizer.pad(
            doc_examples,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors=self.return_tensors)
        if reranker_examples!=None:
            r_collated = self.reranker_tokenizer.pad(
                reranker_examples,
                padding='longest',
                return_attention_mask=True,
                return_tensors='pt'
            )
        #clear tokenizer cache https://github.com/huggingface/tokenizers/pull/1675/
        # self.tokenizer._tokenizer.model._clear_cache()
        # gc.collect()

        # merge into a single BatchEncoding by adding prefix
        for k in list(q_collated.keys()):
            q_collated[q_prefix + k] = q_collated[k]
            del q_collated[k]
        for k in d_collated:
            q_collated[d_prefix + k] = d_collated[k]
        if reranker_examples!=None:
            for k in r_collated:
                q_collated[r_prefix + k] = r_collated[k]

        merged_batch_dict = q_collated
        # dummy placeholder for field "labels", won't use it to compute loss
        labels = torch.zeros(len(query_examples), dtype=torch.long)
        merged_batch_dict['labels'] = labels
        # to identify the source ...
        # YL: hact, q_query_ids are strings, remove to work with transformer 4.37 dataloader
        # merged_batch_dict['q_query_ids']=[ (feature['query_id'] if 'query_id' in feature else str(n))
        #                                      for n, feature in enumerate(features)]

        if 'kd_labels' in features[0]:
            kd_labels = torch.stack([torch.tensor(f['kd_labels']) for f in features], dim=0).float()
            merged_batch_dict['kd_labels'] = kd_labels

        return merged_batch_dict
