import torch

from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import DataCollatorWithPadding, BatchEncoding
from logger_config import logger
from collators.collator_utils import _get_examples
import os


@dataclass
class BiencoderCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, pad_to_multiple_of, tokenizer_teacher = None):
        super(BiencoderCollator, self).__init__(tokenizer, pad_to_multiple_of=pad_to_multiple_of)
        self.tokenizer = tokenizer
        self.tokenizer_teacher = tokenizer_teacher

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:

        # already truncated during tokenization
        q_prefix, d_prefix = 'q_', 'd_'
        teacher_q_prefix, teacher_d_prefix = 'teacher_q_', 'teacher_d_'
        query_examples, doc_examples = _get_examples(q_prefix, d_prefix, features)
        q_collated = self.tokenizer.pad(
            query_examples,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors)
        d_collated = self.tokenizer.pad(
            doc_examples,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors)

        # merge into a single BatchEncoding by adding prefix
        for k in list(q_collated.keys()):
            q_collated[q_prefix + k] = q_collated[k]
            del q_collated[k]
        for k in d_collated:
            q_collated[d_prefix + k] = d_collated[k]

        # based on examples here:https://huggingface.co/intfloat/e5-mistral-7b-instruct
        if self.tokenizer_teacher:
            teacher_query_examples, teacher_doc_examples = _get_examples(teacher_q_prefix, teacher_d_prefix, features)
            teacher_qd_examples = teacher_query_examples + teacher_doc_examples
            pad_to_length = max([len(qd['input_ids']) for qd in teacher_qd_examples])
            teacher_q_collated = self.tokenizer_teacher.pad(
                teacher_query_examples,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors=self.return_tensors)
            teacher_d_collated = self.tokenizer_teacher.pad(
                teacher_doc_examples,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors=self.return_tensors)
            # for k in list(teacher_qd_collated.keys()):
            #     q_collated['teacher_qd_'+ k] = teacher_qd_collated[k]
            for k in list(teacher_q_collated.keys()):
                q_collated[teacher_q_prefix + k] = teacher_q_collated[k]
            for k in teacher_d_collated:
                q_collated[teacher_d_prefix + k] = teacher_d_collated[k]
                # del teacher_d_collated[k]

        merged_batch_dict = q_collated
        # dummy placeholder for field "labels", won't use it to compute loss
        labels = torch.zeros(len(query_examples), dtype=torch.long)
        merged_batch_dict['labels'] = labels
        # to identify the source ...
        merged_batch_dict['q_query_ids']=[ (feature['query_id'] if 'query_id' in feature else str(n))
                                             for n, feature in enumerate(features)]

        if 'kd_labels' in features[0]:
            kd_labels = torch.stack([torch.tensor(f['kd_labels']) for f in features], dim=0).float()
            merged_batch_dict['kd_labels'] = kd_labels

        return merged_batch_dict
