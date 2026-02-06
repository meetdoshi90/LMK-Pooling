import os
import random
from copy import deepcopy
from dataclasses import dataclass

import torch.utils.data.dataset
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import DataCollatorForWholeWordMask
import spacy
import random

from .utils import tensorize_batch


class DatasetForPretraining(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        if os.path.isdir(data_dir):
            datasets = []
            for file in os.listdir(data_dir):
                print(f"Loading {file}")
                file = os.path.join(data_dir, file)
                datasets.append(self.load_dataset(file))
            self.dataset = concatenate_datasets(datasets)
            self.dataset = self.dataset.shuffle(seed=42)
        else:
            print(f"Loading {data_dir}")
            self.dataset = self.load_dataset(data_dir)
        
        print(f"Loaded dataset with {len(self.dataset)} items")

    def load_dataset(self, file):
        if file.endswith('.jsonl') or file.endswith('.json'):
            return load_dataset('json', data_files=file)['train']
        elif os.path.isdir(file):
            return Dataset.load_from_disk(file)
        else:
            raise NotImplementedError(f"Not support this file format:{file}")

    def __getitem__(self, item):
        return self.dataset[item]['text']

    def __len__(self):
        return len(self.dataset)
    
    

# class DatasetForPretrainingLoaded(torch.utils.data.Dataset):
#     def __init__(self, data_dir):
#         self.dataset =  load_from_disk(data_dir)


#     def load_dataset(self, file):
#         if file.endswith('.jsonl') or file.endswith('.json'):
#             return load_dataset('json', data_files=file)['train']
#         elif os.path.isdir(file):
#             return Dataset.load_from_disk(file)
#         else:
#             raise NotImplementedError(f"Not support this file format:{file}")

#     def __getitem__(self, item):
#         return self.dataset[item]['text']

#     def __len__(self):
#         return len(self.dataset)


@dataclass
class RetroMAECollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15

    def __call__(self, examples):
        input_ids_batch = []
        attention_mask_batch = []
        encoder_mlm_mask_batch = []
        decoder_labels_batch = []
        decoder_matrix_attention_mask_batch = []

        for e in examples:

            e_trunc = self.tokenizer.encode(e, max_length=self.max_seq_length, truncation=True)
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]

            self.mlm_probability = self.encoder_mlm_probability
            text_encoder_mlm_mask = self._whole_word_mask(tokens)

            self.mlm_probability = self.decoder_mlm_probability
            mask_set = []
            for _ in range(min(len(tokens), 128)):
                mask_set.append(self._whole_word_mask(tokens))

            text_matrix_attention_mask = []
            for i in range(len(tokens)):
                idx = random.randint(0, min(len(tokens), 128) - 1)
                text_decoder_mlm_mask = deepcopy(mask_set[idx])
                text_decoder_mlm_mask[i] = 1
                text_matrix_attention_mask.append(text_decoder_mlm_mask)

            input_ids_batch.append(torch.tensor(e_trunc))
            attention_mask_batch.append(torch.tensor([1] * len(e_trunc)))
            e_trunc[0] = -100
            e_trunc[-1] = -100
            decoder_labels_batch.append(torch.tensor(e_trunc))

            encoder_mlm_mask_batch.append(torch.tensor(text_encoder_mlm_mask))
            decoder_matrix_attention_mask_batch.append(1 - torch.tensor(text_matrix_attention_mask))

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        origin_input_ids_batch = input_ids_batch.clone()
        encoder_mlm_mask_batch = tensorize_batch(encoder_mlm_mask_batch, 0)
        encoder_input_ids_batch, encoder_labels_batch = self.torch_mask_tokens(input_ids_batch, encoder_mlm_mask_batch)
        decoder_labels_batch = tensorize_batch(decoder_labels_batch, -100)
        matrix_attention_mask_batch = tensorize_batch(decoder_matrix_attention_mask_batch, 0)

        batch = {
            "encoder_input_ids": encoder_input_ids_batch,
            "encoder_attention_mask": attention_mask_batch,
            "encoder_labels": encoder_labels_batch,
            "decoder_input_ids": origin_input_ids_batch,
            "decoder_attention_mask": matrix_attention_mask_batch,  # [B,L,L]
            "decoder_labels": decoder_labels_batch,
        }

        return batch

class LandmarkRetroMAECollator(RetroMAECollator):
    _xx_nlp = spacy.load("xx_sent_ud_sm")
    landmark_token_id = None

    def __post_init__(self):
        super().__post_init__()
        # using seperator token as landmark token
        if self.tokenizer.sep_token_id:
            self.landmark_token_id = self.tokenizer.sep_token_id
        else:
            self.landmark_token_id = self.tokenizer.eos_token_id
        
        if self.tokenizer.cls_token_id:
            self.cls_token_id = self.tokenizer.cls_token_id
        else:
            self.cls_token_id = self.tokenizer.bos_token_id
        
        assert self.landmark_token_id is not None
    
    def __call__(self, examples):
        
        input_ids_batch = []
        attention_mask_batch = []
        encoder_mlm_mask_batch = []
        decoder_labels_batch = []
        decoder_matrix_attention_mask_batch = []

        for e in examples:
            e_trunc = self.get_tokenized_seq(e)

            # e_trunc = self.tokenizer.encode(e, max_length=self.max_seq_length, truncation=True)
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]

            self.mlm_probability = self.encoder_mlm_probability
            text_encoder_mlm_mask = self._whole_word_mask(tokens)
            # dont mask separator/landmark tokens
            for i,tid in enumerate(e_trunc):
                if tid == self.landmark_token_id:
                    text_encoder_mlm_mask[i] = 0

            self.mlm_probability = self.decoder_mlm_probability
            mask_set = []
            for _ in range(min(len(tokens), 128)):
                mask_set.append(self._whole_word_mask(tokens))

            text_matrix_attention_mask = []
            for i in range(len(tokens)):
                idx = random.randint(0, min(len(tokens), 128) - 1)
                text_decoder_mlm_mask = deepcopy(mask_set[idx])
                text_decoder_mlm_mask[i] = 1
                text_matrix_attention_mask.append(text_decoder_mlm_mask)

            input_ids_batch.append(torch.tensor(e_trunc))
            attention_mask_batch.append(torch.tensor([1] * len(e_trunc)))
            e_trunc[0] = -100
            e_trunc[-1] = -100
            # exclude landmark tokens from decoder labels
            for i,tid in enumerate(e_trunc):
                if tid == self.landmark_token_id:
                    e_trunc[i] = -100

            decoder_labels_batch.append(torch.tensor(e_trunc))

            encoder_mlm_mask_batch.append(torch.tensor(text_encoder_mlm_mask))
            decoder_matrix_attention_mask_batch.append(1 - torch.tensor(text_matrix_attention_mask))

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        origin_input_ids_batch = input_ids_batch.clone()
        encoder_mlm_mask_batch = tensorize_batch(encoder_mlm_mask_batch, 0)
        encoder_input_ids_batch, encoder_labels_batch = self.torch_mask_tokens(input_ids_batch, encoder_mlm_mask_batch)
        decoder_labels_batch = tensorize_batch(decoder_labels_batch, -100)
        matrix_attention_mask_batch = tensorize_batch(decoder_matrix_attention_mask_batch, 0)

        batch = {
            "encoder_input_ids": encoder_input_ids_batch,
            "encoder_attention_mask": attention_mask_batch,
            "encoder_labels": encoder_labels_batch,
            "decoder_input_ids": origin_input_ids_batch,
            "decoder_attention_mask": matrix_attention_mask_batch,  # [B,L,L]
            "decoder_labels": decoder_labels_batch,
        }

        return batch
    
    def get_tokenized_seq(self, e):
        # add the landmark tokens after splitting sentence
        # splitted_text = self.sentence_splitter.split(text=e)
        with self._xx_nlp.memory_zone():
            doc = self._xx_nlp(e)
            splitted_text = [sent.text for sent in doc.sents]
        splitted_text = [t for t in splitted_text if t and t.strip()] # filter blank strings
        e_trunc = self.tokenizer(
                splitted_text,
                padding=False,
                truncation=False,
                add_special_tokens=False,
                return_attention_mask=False
        )['input_ids']
        # Now we will concatenate the tokenized sentences in [CLS] + S1 + [LMK] + S2 ... [LMK]
        e_trunc = [x + [self.landmark_token_id] for x in e_trunc]
        e_trunc = ([self.cls_token_id] + [x for sub in e_trunc for x in sub])[:self.max_seq_length]
        # if sentence if too long that it does not get split into LMK then add an LMK manually at the end cut-off
        if self.landmark_token_id not in e_trunc:
            e_trunc[-1] = self.landmark_token_id
        
        return e_trunc


class FixedChunkLandmarkRetroMAECollator(LandmarkRetroMAECollator):
    sentence_splitter = None
    landmark_token_id = None
    landmark_chunk_size = 256
    
    def get_tokenized_seq(self, e):
        # add the landmark tokens after self.landmark_chunk_size tokens
        tokenized = self.tokenizer.encode(e, truncation=False)
        splitted_tokens = [tokenized[i:i + self.landmark_chunk_size] for i in range(0, len(tokenized), self.landmark_chunk_size)]
        # Now we will concatenate the tokenized chunks in [CLS] + S1 + [LMK] + S2 ... [LMK]
        e_trunc = [x + [self.landmark_token_id] for x in splitted_tokens]
        e_trunc = ([self.cls_token_id] + [x for sub in e_trunc for x in sub])[:self.max_seq_length]
        # if sentence if too long that it does not get split into LMK then add an LMK manually at the end cut-off
        if self.landmark_token_id not in e_trunc:
            e_trunc[-1] = self.landmark_token_id
        
        return e_trunc

class VariableChunkLandmarkRetroMAECollator(FixedChunkLandmarkRetroMAECollator):
    landmark_chunk_sizes = [32, 64, 128, 256]
    
    def get_tokenized_seq(self, e):
        # add the landmark tokens after self.landmark_chunk_size tokens
        tokenized = self.tokenizer.encode(e, truncation=False)
        splitted_tokens = []
        i=0
        while i < len(tokenized):
            # create chunks of variable sizes
            chunk_size = random.choice(self.landmark_chunk_sizes)
            # Add the next chunk of tokens
            chunk_end = min(i + chunk_size, len(tokenized))
            splitted_tokens.append(tokenized[i:chunk_end])
            i = chunk_end

        # Now we will concatenate the tokenized chunks in [CLS] + S1 + [LMK] + S2 ... [LMK]
        e_trunc = [x + [self.landmark_token_id] for x in splitted_tokens]
        e_trunc = ([self.cls_token_id] + [x for sub in e_trunc for x in sub])[:self.max_seq_length]
        # if sentence if too long that it does not get split into LMK then add an LMK manually at the end cut-off
        if self.landmark_token_id not in e_trunc:
            e_trunc[-1] = self.landmark_token_id
        
        return e_trunc