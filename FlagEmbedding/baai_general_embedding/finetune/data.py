import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple
import json
import datasets
from datasets import interleave_datasets
from torch.utils.data import Dataset,IterableDataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding

from .arguments import DataArguments

import pdb

class TrainDatasetForEmbedding(IterableDataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        self.tokenizer = tokenizer
        self.args = args
        if os.path.isdir(args.train_data):
            # train_datasets = []
            # for file in os.listdir(args.train_data):
            #     temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
            #                                          split='train')
            #     if len(temp_dataset) > args.max_example_num_per_dataset:
            #         temp_dataset = temp_dataset.select(
            #             random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
            #     train_datasets.append(temp_dataset)
            # train_datasets_probs = self.get_probs(args.train_data_config)
            # self.dataset = interleave_datasets(train_datasets, probabilities=train_datasets_probs, stopping_strategy="all_exhausted", seed=42)
            self.dataset = self.get_dataset(args.train_data,args.train_data_config)
            #self.dataset = datasets.concatenate_datasets(train_datasets)
            # print(len(self.dataset))
            
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        
        #self.total_len = len(self.dataset)
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        return iter(self.dataset)

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        query = self.dataset[item]['query']
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_for_retrieval + query
        passages = []
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)
        if not self.args.in_batch_neg:
            #print("Data has hard negatives as well")
            if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
                num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
                negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
            else:
                negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
            passages.extend(negs)
        # else:
        #     print("Training using in-batch negatives")

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [self.args.passage_instruction_for_retrieval+p for p in passages]
        return query, passages
    def get_dataset(self,data_dir, config_file):
        with open(config_file, 'r') as f:
            data_config = json.load(f)
        train_datasets_probs = []
        train_datasets_weight = []
        train_datasets = []
        for dc in data_config:
            print("Loading dataset ",dc['name'])
            ds = datasets.load_dataset('json', data_files=data_dir+dc['name'], split='train',streaming=True)
            if self.args.query_instruction_for_retrieval is not None:
                tmp_ds =  ds.map(lambda example: {'query': self.args.query_instruction_for_retrieval  + example['query']})
                train_datasets.append(tmp_ds)
            else:
                train_datasets.append(ds)
            train_datasets_weight.append(dc['weight'])
        train_datasets_probs=[w/sum(train_datasets_weight) for w in train_datasets_weight]
        final_train_dataset = interleave_datasets(train_datasets, probabilities=train_datasets_probs, stopping_strategy="all_exhausted", seed=42)

        return final_train_dataset.with_format("torch")
    
    
class EvalDatasetForEmbedding(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        if os.path.isdir(args.eval_data):
            eval_datasets = []
            for file in os.listdir(args.eval_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.eval_data, file),
                                                     split='train')
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                eval_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(eval_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.eval_data, split='train')

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        query = self.dataset[item]['query']
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_for_retrieval + query

        passages = []
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)

        # if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
        #     num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
        #     negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        # else:
        #     negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        # passages.extend(negs)

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [self.args.passage_instruction_for_retrieval+p for p in passages]
        return query, passages



@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 512
    passage_max_len: int = 512

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
            passage = [f['pos'] for f in features]
        except:
            query = [f[0] for f in features]
            passage = [f[1] for f in features]
            

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])
        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated}
    
    
@dataclass
class ScoringCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 64
    passage_max_len: int = 256

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
        query = [f[0] for f in features]
        passage = [f[1] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated,"q_orig":query,"p_orig":passage}
