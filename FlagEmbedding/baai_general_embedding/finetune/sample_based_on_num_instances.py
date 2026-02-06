import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple
import json
import datasets
from torch.utils.data import Dataset
import glob
from tqdm import tqdm
def get_probs():
    num_lines_list = []
    names =  []
    data_prob ={}
    files_list = glob.glob("/dccstor/retrieve-rerank2/irl/data/unsup/*.jsonl")
    name_list = [i.split("/")[-1]for i in files_list]

    for file_name,name in tqdm(zip(files_list,name_list)):
        num_lines = sum(1 for _ in open(file_name))
        names.append(name)
        num_lines_list.append(num_lines)
    prob_list = [1-(i/sum(num_lines_list)) for i in num_lines_list]
    prob_list_final = [i/sum(prob_list) for i in prob_list]
    
    for item,name in zip(prob_list_final,names):
        data_prob[name] = item
    
    
    return data_prob
train_datasets = []
#data_prob_dict = {'nq_full_with_hard_negatives.jsonl': 0.4124629080118694, 'stackexchange_with_hard_negatives.jsonl': 0.5341246290801187, 'stackexchange_title_body_economics.stackexchange.com_with_hn.jsonl': 0.0009891196834817012, 'stackexchange_TitleBody_Answer_quant.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_Title_Answer_quant.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_TitleBody_Answer_philosophy.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_Title_Answer_philosophy.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_title_body_philosophy.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_TitleBody_Answer_law.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_Title_Answer_law.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_title_body_quant.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_title_body_law.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_TitleBody_Answer_crypto.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_Title_Answer_crypto.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_TitleBody_Answer_bitcoin.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_Title_Answer_bitcoin.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_title_body_crypto.stackexchange.com_with_hn.jsonl': 0.0019782393669634025, 'stackexchange_title_body_bitcoin.stackexchange.com_with_hn.jsonl': 0.002967359050445104, 'stackexchange_TitleBody_Answer_money.stackexchange.com_with_hn.jsonl': 0.002967359050445104, 'stackexchange_Title_Answer_money.stackexchange.com_with_hn.jsonl': 0.002967359050445104, 'stackexchange_Title_Answer_unix.stackexchange.com_with_hn.jsonl': 0.01582591493570722}
data_prob_dict = get_probs()
json.dump(data_prob_dict,open("./data_prob_dict.json","w"))
print(data_prob_dict)
# probabilities = []
# seed = 42
# for file in sorted(os.listdir(args.train_data)):
#     probabilities.append(data_prob_dict[file])
#     temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),split='train')
#     train_datasets.append(temp_dataset)

    
#     # if len(temp_dataset) > args.max_example_num_per_dataset:
#     #     temp_dataset = temp_dataset.select(
#     #         random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
#     # train_datasets.append(temp_dataset)
# self.dataset = datasets.interleave_datasets(train_datasets,probabilities=probabilities, seed=seed, stopping_strategy="all_exhausted")
# #self.dataset = datasets.concatenate_datasets(train_datasets)
# else:
# self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')