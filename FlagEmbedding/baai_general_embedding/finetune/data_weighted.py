import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple
import json
import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding

from .arguments import DataArguments


def get_probs():
    # data = json.load(open("/dccstor/cssblr/vishwajeet/git/2023/HyDE/data_nq_finance_unix.json"))
    # weights = []
    # names =  []
    # data_prob ={}
    # for d in data:
    #     if ".jsonl.gz" in d["name"]:
    #         names.append(d["name"].replace(".jsonl.gz","").replace(".com",".com_with_hn").replace("/","_")+".jsonl")
    #         weights.append(d["weight"])
    #     else:
    #         names.append(d["name"])
    #         weights.append(d["weight"])
                         
    # probs = []
    # for item,name in zip(weights,names):
    #     data_prob[name] = item/sum(weights)
    # print(data_prob)
    data_prob = json.load(open("/dccstor/cssblr/vishwajeet/git/2023/FlagEmbedding/data_prob_dict.json"))
    return data_prob

class WeightedTrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        
        if os.path.isdir(args.train_data):
            train_datasets = []
            #data_prob_dict = {'nq_full_with_hard_negatives.jsonl': 0.4124629080118694, 'stackexchange_with_hard_negatives.jsonl': 0.5341246290801187, 'stackexchange_title_body_economics.stackexchange.com_with_hn.jsonl': 0.0009891196834817012, 'stackexchange_TitleBody_Answer_quant.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_Title_Answer_quant.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_TitleBody_Answer_philosophy.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_Title_Answer_philosophy.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_title_body_philosophy.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_TitleBody_Answer_law.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_Title_Answer_law.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_title_body_quant.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_title_body_law.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_TitleBody_Answer_crypto.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_Title_Answer_crypto.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_TitleBody_Answer_bitcoin.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_Title_Answer_bitcoin.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_title_body_crypto.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_title_body_bitcoin.stackexchange.com_with_hn.jsonl': 0.002967359050445104, 'stackexchange_TitleBody_Answer_money.stackexchange.com_with_hn.jsonl': 0.002967359050445104, 'stackexchange_Title_Answer_money.stackexchange.com_with_hn.jsonl': 0.002967359050445104, 'stackexchange_Title_Answer_unix.stackexchange.com_with_hn.jsonl': 0.01582591493570722}
            data_prob_dict = get_probs()
            probabilities = []
            seed = 42
            for file in sorted(os.listdir(args.train_data)):
                probabilities.append(data_prob_dict[file])
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),split='train')
                train_datasets.append(temp_dataset)

                
                # if len(temp_dataset) > args.max_example_num_per_dataset:
                #     temp_dataset = temp_dataset.select(
                #         random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                # train_datasets.append(temp_dataset)
            self.dataset = datasets.interleave_datasets(train_datasets,probabilities=probabilities, seed=seed)
            #self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

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

        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        passages.extend(negs)

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [self.args.passage_instruction_for_retrieval+p for p in passages]
        return query, passages

class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

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

        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        passages.extend(negs)

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
