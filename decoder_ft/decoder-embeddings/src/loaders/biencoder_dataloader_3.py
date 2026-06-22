import os
import random

from typing import Tuple, Dict, List, Optional
from datasets import load_dataset, DatasetDict, Dataset, interleave_datasets
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedTokenizerFast, Trainer
import torch

from . import RetrievalDataLoader2
from config import Arguments
from logger_config import logger
from .loader_utils import group_doc_ids, get_dataset_attrs_from_filename
import json
from datasets.distributed import split_dataset_by_node
import torch.distributed as dist


class RetrievalDataLoader3 (RetrievalDataLoader2):

    def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.negative_size = args.train_n_passages - 1
        #assert self.negative_size > 0
        self.tokenizer = tokenizer
        #corpus_path = os.path.join(args.data_dir, 'passages.jsonl.gz')
        #self.corpus: Dataset = load_dataset('json', data_files=corpus_path)['train']
        (self.train_dataset,   # the element-wise interleaved dataset
         self.eval_dataset, 
         self.base_dataset_names,  # names of underlying datasets
         self.base_dataset_probs,  # and their probs
         self.base_datasets         # and the underying datasets
         ) = self._get_transformed_datasets()

        # use its state to decide which positives/negatives to sample
        self.trainer: Optional[Trainer] = None
        logger.info('end of RetrievalDataLoader3.__init__() %s', os.getpid())


    def _transform_func_notitle(self, examples: Dict[str, List]) -> Dict[str, List]:
        try:  # TODO - is this the best way?
            current_epoch = int(self.trainer.state.epoch or 0)
        except:
            current_epoch = 0

        input_doc_ids: List[int] = group_doc_ids(
            examples=examples,
            negative_size=self.negative_size,
            offset=random.randint(0,60)+ self.args.seed, #randomly pick positive/negatives, if there are multiple ones
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
            all_doc_ids.extend(examples['negatives'][i]['doc_id'])
            all_doc_texts.extend(examples['positives'][i]['docs'])
            all_doc_texts.extend(examples['negatives'][i]['docs'])
            all_doc_titles.extend(examples['positives'][i]['titles'])
            all_doc_titles.extend(examples['negatives'][i]['titles'])
        all_doc_ids = [ int(d) for d in all_doc_ids ]
        doc_text = dict(zip(all_doc_ids,all_doc_texts))
        doc_titles = dict(zip(all_doc_ids,all_doc_titles))
        
        input_docs: List[str] = [doc_text[doc_id] for doc_id in input_doc_ids]
        input_titles: List[str] = [doc_titles[doc_id] for doc_id in input_doc_ids]

        query_batch_dict = self.tokenizer(examples['query'],
                                          max_length=self.args.q_max_len,
                                          padding=PaddingStrategy.DO_NOT_PAD,
                                          truncation=True)
        doc_batch_dict = self.tokenizer(input_docs,
                                        max_length=self.args.p_max_len,
                                        padding=PaddingStrategy.DO_NOT_PAD,
                                        truncation=True)

        merged_dict = {'q_{}'.format(k): v for k, v in query_batch_dict.items()}
        step_size = self.args.train_n_passages
        for k, v in doc_batch_dict.items():
            k = 'd_{}'.format(k)
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
                    qid_to_doc_id_to_score[q_id][int(doc_id)] = score

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

        # Custom formatting function must return a dict
        return merged_dict




    def _get_transformed_datasets(self) -> Tuple:
        # YL: simplified code to only use train_data_config and
        # val_data_config to load training and validation data
        streaming=True
        
        config_file = self.args.train_data_config
        data_dir= self.args.data_dir
        # Load the json dataset config  file
        train_datasets = []
        train_datasets_weight = []
        train_datasets_name = []
        with open(config_file, 'r') as f:
            data_config = json.load(f)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        # print(rank, world_size)
        for dc in data_config:
            logger.info(f"Loading dataset {dc['name']}")
            ds = load_dataset('json', data_files=[data_dir+dc['name']], streaming=streaming)['train']
            ds = split_dataset_by_node(ds, world_size=world_size, rank=rank)
            # if streaming:
            #     ds = ds.shuffle(buffer_size=50000, seed=self.args.seed)
            train_datasets_weight.append(dc['weight'])
            train_datasets_name.append(dc['name'])
            # The collator reads _instruction / _max_length_q / _max_length_p directly
            # from the example, eliminating the broken query_id[:2] lookup.
            attrs = get_dataset_attrs_from_filename(dc['name'])
            _instr   = attrs["instruction"]
            _max_q   = attrs["max_length_q"]
            _max_p   = attrs["max_length_p"]
            ds = ds.map(lambda ex: {
                **ex,
                "_instruction":  _instr,
                "_max_length_q": _max_q,
                "_max_length_p": _max_p,
            })
            logger.info(f'Instruction for dataset {dc["name"]} = {_instr}')

            # TODO - this is supposed to be total, not per dataset?
            if self.args.max_train_samples is not None and not isinstance(ds, torch.utils.data.IterableDataset):
                ds = ds.select(range(self.args.max_train_samples))
            # if self.args.no_titles:
            #     ds = ds.map(self._transform_func_notitle, batched=True, batch_size=self.args.train_batch_size)
            # else:
            #     ds = ds.map(self._transform_func, batched=True, batch_size=self.args.train_batch_size)
            train_datasets.append(ds)

        logger.info(f"Sum of weights {sum(train_datasets_weight)}")
        train_datasets_prob=[w/sum(train_datasets_weight) for w in train_datasets_weight]
        eval_dataset={}
        if self.args.do_eval:
            assert self.args.val_data_config, "pelase provide validation data config file for configuration"
            val_config_file = self.args.val_data_config
            with open(val_config_file, 'r') as f:
                val_data_config = json.load(f)
            for dc in val_data_config:
                logger.info(f"Loading dataset {dc['name']}")
                ds = load_dataset('json', data_files=[data_dir+dc['name']], streaming=streaming)['train']
                # if self.args.no_titles:
                #     ds = ds.map(self._transform_func_notitle, batched=True, batch_size=self.args.eval_batch_size)
                # else:
                #     ds = ds.map(self._transform_func, batched=True, batch_size=self.args.eval_batch_size)
                eval_dataset[dc['name']] = ds

        return None, eval_dataset, train_datasets_name, train_datasets_prob, train_datasets

    # def _get_transformed_datasets(self) -> Tuple:
    #     streaming=True
    #     data_files = {}
    #     if self.args.train_file is not None:
    #         data_files["train"] = self.args.train_file.split(',')
    #     if self.args.validation_file is not None:
    #         data_files["validation"] = self.args.validation_file
    #     #raw_datasets: DatasetDict = load_dataset('json', data_files=data_files)
        
    #     #The train config file should come from args. Hard coded for now.
    #     config_file = self.args.train_data_config
    #     data_dir= self.args.data_dir
    #     # Load the json dataset config  file
    #     train_datasets = []
    #     train_datasets_weight = []
    #     train_datasets_name = []
    #     with open(config_file, 'r') as f:
    #         data_config = json.load(f)
    #     for dc in data_config:
    #         logger.info(f"Loading dataset {dc['name']}")
    #         ds = load_dataset('json', data_files=[data_dir+dc['name']], streaming=streaming)['train']
    #         ds = ds.shuffle(buffer_size=50000, seed=self.args.seed)
    #         train_datasets_weight.append(dc['weight'])
    #         train_datasets_name.append(dc['name'])

    #         # TODO - this is supposed to be total, not per dataset?
    #         if self.args.max_train_samples is not None and not isinstance(ds, torch.utils.data.IterableDataset):
    #             ds = ds.select(range(self.args.max_train_samples))
    #         if self.args.no_titles:
    #             ds = ds.map(self._transform_func_notitle, batched=True, batch_size=self.args.train_batch_size)
    #         else:
    #             ds = ds.map(self._transform_func, batched=True, batch_size=self.args.train_batch_size)
    #         train_datasets.append(ds)


    #     train_datasets_prob=[w/sum(train_datasets_weight) for w in train_datasets_weight]

    #     raw_datasets = DatasetDict()
    #     # raw_datasets['train'] = interleave_datasets(train_datasets, probabilities=train_datasets_prob, stopping_strategy="all_exhausted", seed=42)
    #     # # make val dataset streaming to try to prevent "tokenizer before fork" warning - didn't seem to work
    #     # raw_datasets['validation'] = load_dataset('json', data_files=data_files['validation'], streaming=streaming)['train']
        
    #     train_dataset, eval_dataset = None, None
        
    #     if self.args.do_eval:
    #         if "validation" not in raw_datasets:
    #             raise ValueError("--do_eval requires a validation dataset")
    #         eval_dataset = raw_datasets["validation"]
    #         eval_dataset.set_transform(self._transform_func)

    #     return train_dataset, eval_dataset, train_datasets_name, train_datasets_prob, train_datasets
