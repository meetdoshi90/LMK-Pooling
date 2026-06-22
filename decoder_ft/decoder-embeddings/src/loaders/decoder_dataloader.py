import os
import random

from typing import Tuple, Dict, List, Optional
from datasets import load_dataset, DatasetDict, Dataset, interleave_datasets
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedTokenizerFast, Trainer
import torch

from . import RetrievalDataLoader3
from config import Arguments
from logger_config import logger
from .loader_utils import group_doc_ids, dataset_attributes, insert_lmk_tokens, sample_lmk_granularity
import json
from collections import deque
import psutil
import gc
import torch.distributed as dist

class RetrievalDecoderDataLoader(RetrievalDataLoader3):


    def __init__(self, args, tokenizer, reranker_tokenizer):
        self.args              = args
        self.negative_size     = args.train_n_passages - 1
        self.tokenizer         = tokenizer
        self.reranker_tokenizer = reranker_tokenizer

        # ── LMK setup ────────────────────────────────────────────────────────
        self._use_lmk      = (getattr(args, 'pooling_source', '') == 'lmk')
        self._lmk_gran     = getattr(args, 'lmk_granularity', 64)
        raw_set            = getattr(args, 'lmk_granularity_set', None)
        self._lmk_gran_set = (
            [int(g.strip()) for g in raw_set.split(',') if g.strip()]
            if raw_set else None
        )

        (self.train_dataset,
         self.eval_dataset,
         self.base_dataset_names,
         self.base_dataset_probs,
         self.base_datasets) = self._get_transformed_datasets()

        self.trainer = None
        logger.info('end of RetrievalDecoderDataLoader.__init__() %s', os.getpid())

    def _tokenize_with_lmk_or_eos(
        self,
        texts: List[str],
        max_length: int,
    ) -> List[List[int]]:
        """
        For LMK: tokenize → insert EOS landmarks every N tokens.
        For mean/last: tokenize(max_length-1) → append EOS (original behaviour).
        """
        eos_id = self.tokenizer.eos_token_id
        if self._use_lmk:
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

    def convert_to_bge_format(self, examples):
        # print(examples)
        query = examples['query']
        positives = examples['pos']
        negatives = examples['neg']
        positive_scores = examples['pos_scores']
        negative_scores = examples['neg_scores']
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
        return new_examples

    def _transform_func_notitle(self, examples: Dict[str, List]) -> Dict[str, List]:
        try:  # TODO - is this the best way?
            current_epoch = int(self.trainer.state.epoch or 0)
        except:
            current_epoch = 0
        # process = psutil.Process()
        # mem_info = process.memory_info()
        # print("Total batched samples", total_batched_samples)
        # print(f"Current PID: {psutil.Process().pid}", f"RSS: {mem_info.rss/1e6:.2f} MB, VMS: {mem_info.vms/1e6:.2f} MB")
        rotate_positive_as_negative = False

        # rank = dist.get_rank()
        # if rank==0:
        #     print(examples['query'])
        if 'pos' in examples:
            # examples = self.convert_to_bge_format(examples)
            query = examples['query']
            positives = examples['pos']
            negatives = examples['neg']
            positive_scores = examples['pos_scores']
            negative_scores = examples['neg_scores']
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
            negative_size=self.negative_size,
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
            if self.negative_size:
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
            step_size = self.args.train_n_passages
            for i, q in enumerate(examples['query']):
                for d in input_docs[i*step_size:(i+1)*step_size]:
                    reranker_batch_text.append([q,d])
            reranker_batch_dict = self.reranker_tokenizer(
                reranker_batch_text,
                max_length=self.reranker_tokenizer.config.model_max_length,
                padding=PaddingStrategy.DO_NOT_PAD,
                return_attention_mask=False,
                truncation=True
            )
        else:
            reranker_batch_dict = None

        def _get_detailed_instruct(task_description: str, query: str) -> str:
            return f'Instruct: {task_description}\nQuery: {query}'
        
        # Replace the data_id lookup block with:
        if '_instruction' in examples:
            task  = examples['_instruction'][0]
            max_length_q = examples['_max_length_q'][0]
            max_length_p = examples['_max_length_p'][0]
        else:
            task  = 'Given a web search query, retrieve relevant passages that answer the query'
            max_length_q = self.args.q_max_len
            max_length_p = self.args.p_max_len

        for i, q in enumerate(examples['query']):
            q_t = _get_detailed_instruct(task, q)
            examples['query'][i] = q_t

        query_batch_dict = {'input_ids': self._tokenize_with_lmk_or_eos(examples['query'], max_length_q)}
        doc_batch_dict   = {'input_ids': self._tokenize_with_lmk_or_eos(input_docs,        max_length_p)}

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
                    merged_dict[k] = []
                    for idx in range(0, len(v), step_size):
                        merged_dict[k].append(v[idx:(idx + step_size)])

            if self.args.do_kd_biencoder:
                qid_to_doc_id_to_score = {}

                def _update_qid_pid_score(q_id: str, ex: Dict):
                    assert len(ex['doc_id']) == len(ex['score'])
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

        # gc.collect()
        # Custom formatting function must return a dict
        return merged_dict
    