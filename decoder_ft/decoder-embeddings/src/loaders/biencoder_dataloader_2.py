import os
import random

from typing import Tuple, Dict, List, Optional
from datasets import load_dataset, DatasetDict, Dataset, interleave_datasets
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedTokenizerFast, Trainer

from config import Arguments
from logger_config import logger
from .loader_utils import group_doc_ids
import json


class RetrievalDataLoader2:

    def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.negative_size = args.train_n_passages - 1
        #assert self.negative_size > 0
        self.tokenizer = tokenizer
        #corpus_path = os.path.join(args.data_dir, 'passages.jsonl.gz')
        #self.corpus: Dataset = load_dataset('json', data_files=corpus_path)['train']
        self.train_dataset, self.eval_dataset = self._get_transformed_datasets()

        # use its state to decide which positives/negatives to sample
        self.trainer: Optional[Trainer] = None

    def _transform_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        try:  # TODO - is this the best way?
            current_epoch = int(self.trainer.state.epoch or 0)
        except:
            current_epoch = 0

        input_doc_ids: List[int] = group_doc_ids(
            examples=examples,
            negative_size=self.negative_size,
            offset=current_epoch + self.args.seed,
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
        doc_batch_dict = self.tokenizer(input_titles,
                                        text_pair=input_docs,
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
        data_files = {}
        if self.args.train_file is not None:
            data_files["train"] = self.args.train_file.split(',')
        if self.args.validation_file is not None:
            data_files["validation"] = self.args.validation_file
        #raw_datasets: DatasetDict = load_dataset('json', data_files=data_files)
        
        #The train config file should come from args. Hard coded for now.
        config_file = self.args.train_data_config
        data_dir= self.args.data_dir
        # Load the json dataset config  file
        train_datasets = []
        train_datasets_weight = []
        with open(config_file, 'r') as f:
            data_config = json.load(f)
        for dc in data_config:
            logger.info(f"Loading dataset {dc['name']}")
            ds = load_dataset('json', data_files=[data_dir+dc['name']], streaming=True)['train']
            train_datasets.append(ds)
            train_datasets_weight.append(dc['weight'])
        train_datasets_prob=[w/sum(train_datasets_weight) for w in train_datasets_weight]

        raw_datasets = DatasetDict()
        raw_datasets['train'] = interleave_datasets(train_datasets, probabilities=train_datasets_prob, stopping_strategy="all_exhausted", seed=42)
        raw_datasets['validation'] = load_dataset('json', data_files=data_files['validation'])['train']
        
        train_dataset, eval_dataset = None, None

        if self.args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if self.args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.args.max_train_samples))
            
            train_dataset = train_dataset.map(self._transform_func, batched=True, batch_size=self.args.train_batch_size)
        
        if self.args.do_eval:
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            eval_dataset.set_transform(self._transform_func)

        return train_dataset, eval_dataset
