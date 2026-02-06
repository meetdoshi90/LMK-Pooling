import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple
import json
import numpy as np
import datasets
from datasets import interleave_datasets, Features, Value, Sequence
from torch.utils.data import Dataset,IterableDataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding
import torch
from .arguments import DataArguments
import torch.distributed as dist
import pdb


def get_non_streaming_sharded_dataset(path, seed):
        # load
        features = Features({
            "query":      Value("string"),
            "pos":        Sequence(Value("string")),
            "neg":        Sequence(Value("string")),
            "weight":        Value("float64"), #for crisp
            # make scores optional by declaring them here…
            "pos_scores": Sequence(Value("float64")),
            "neg_scores": Sequence(Value("float64")),
        })
        ds = datasets.load_dataset("json", data_files=path, split='train',streaming=False, features=features)
        # shard per-GPU
        n, r = (dist.get_world_size(), dist.get_rank()) if dist.is_initialized() else (1, 0)
        print(n,r)
        ds_len = len(ds)
        if ds_len>n:
            ds = ds.shard(num_shards=n, index=r)
        # shuffle within shard
        # ds = ds.shuffle(buffer_size=50000, seed=seed)
        return ds

class DatasetHandler():
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer,
            is_train_data: bool = True
    ):
        self.tokenizer = tokenizer
        self.args = args
        if is_train_data:
            if os.path.isdir(args.train_data):
                self.train_datasets,self.train_datasets_weight,self.train_datasets_probs,self.train_datasets_names,self.train_datasets_psis, self.train_datasets_num_instances = self.get_dataset(args.train_data,args.train_data_config,is_train_data)
                print('#'*20)
                print('Train dataset names and probs and total size')
                print(self.train_datasets_probs)
                print(self.train_datasets_names)
                print(self.train_datasets_num_instances)
            else:
                self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')
            
        else:
            if args.eval_data == None:
                self.eval_datasets = self.eval_datasets_weight = self.eval_datasets_probs = self.eval_datasets_names = self.eval_datasets_psis = self.eval_datasets_num_instances = None
            elif os.path.isdir(args.eval_data):
                self.eval_datasets,self.eval_datasets_weight,self.eval_datasets_probs,self.eval_datasets_names,self.eval_datasets_psis, self.eval_datasets_num_instances = self.get_dataset(args.eval_data,args.eval_data_config,is_train_data)
                print('#'*20)
                print('Eval dataset names and probs')
                print(self.eval_datasets_probs)
                print(self.eval_datasets_names)
                print(self.eval_datasets_num_instances)
            else:
                self.dataset = datasets.load_dataset('json', data_files=args.eval_data, split='train')

    
    
    def initialize_seed(self):
        base_seed = 1235  # Define your base seed
        rank = 0  # Default rank for non-distributed setups

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

        seed = base_seed + rank
        print(f"Initialized seed {seed} for rank {rank}")
        return seed
    
    def calculate_dataset_probs(self,train_datasets_num_instances):
        total = sum(train_datasets_num_instances)
        p = np.array([item*(1.0/total) for item in train_datasets_num_instances])
        p_temp = np.power(p, 1.0/5.0)
        p_temp = p_temp / np.sum(p_temp)
        return p_temp
    
    
        
    def get_dataset(self,data_dir, config_file, is_train_data=True):
        with open(config_file, 'r') as f:
            data_config = json.load(f)
        train_datasets_probs = []
        train_datasets_weight = []
        train_datasets_num_instances = []
        train_datasets = []
        train_dataset_names = []
        seed = self.initialize_seed()
        print('Using seed', seed)
        for dc in data_config:
            print("Loading dataset ",dc['name'])
            train_dataset_names.append(dc["name"])
            if not self.args.is_train_data_streaming and is_train_data:
                ds = get_non_streaming_sharded_dataset(data_dir+dc['name'], seed=seed)
            else:
                ds = datasets.load_dataset('json', data_files=data_dir+dc['name'], split='train',streaming=True)
                if is_train_data:
                    ds = ds.shuffle(buffer_size=50000, seed=seed)
            train_datasets.append(ds)
            if is_train_data:
                if self.args.train_datasets_sampling_init=='weight':
                    print('Init of train datasets has been done with weights')
                    train_datasets_weight.append(dc['weight'])
                elif self.args.train_datasets_sampling_init=='lines':
                    print('Init of train datasets has been done with lines')
                    train_datasets_weight.append(dc['lines'])
                else:
                    raise Exception('Init not mentioned for train datasets')
            else:
                train_datasets_weight.append(1) #Equal weights for all val datasets
            train_datasets_num_instances.append(dc['lines'])
        
        train_datasets_weight = [pow(x,1.0/self.args.train_datasets_sampling_temp) for x in train_datasets_weight]       
        train_datasets_probs=[w/sum(train_datasets_weight) for w in train_datasets_weight]
        train_datasets_psis = [np.log(x) for x in train_datasets_probs]
        
        return train_datasets,train_datasets_weight,train_datasets_probs,train_dataset_names, train_datasets_psis, sum(train_datasets_num_instances)
    
    
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

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [self.args.passage_instruction_for_retrieval+p for p in passages]
        return query, passages



