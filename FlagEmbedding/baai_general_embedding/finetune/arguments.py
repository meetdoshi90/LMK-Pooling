import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    infonce: bool = field(default=False,metadata={"help":"Use infonce loss"}
                          )
    normlized: bool = field(default=True)
    disable_rope: bool = field(default=False)
    
    lora: bool = field(default=False,metadata={"help":"use LORA FT or not"})
    


@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_data_config: str = field(
        default=None, metadata={"help": "Path to train data configuration file"}
    )
    train_datasets_sampling_init: str = field(default='weight', metadata={"help": "Init weights of training datasets to take weight or lines"})
    train_datasets_sampling_temp: float = field(default=1.0)
    eval_data: str = field(
        default=None, metadata={"help": "Path to eval data"}
    )
    eval_data_config: str = field(
        default=None, metadata={"help": "Path to eval data configuration file"}
    )
    train_group_size: int = field(default=8)
    is_train_data_streaming: bool = field(default=True)
    is_neg_distil_collator: bool = field(default=False)

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "the max number of examples for each dataset"}
    )

    query_instruction_for_retrieval: str= field(
        default=None, metadata={"help": "instruction for query"}
    )
    passage_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for passage"}
    )
    in_batch_neg: bool = field(
        default=False, metadata={"help": "use only in batch negatives"}
    )
    interleave: bool = field(
        default=False, metadata={"help": "use interleave datasets"}
    )
    
    # def __post_init__(self):
    #     if not os.path.exists(self.train_data) or os.path.exists(self.eval_data) :
    #         raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(default=False, metadata={"help": "Freeze the parameters of position embeddings"})
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls/mean/lmk_en/lmk_multi/lmk_fixed/lmk_var/latent_attn/every_64"})
    landmark_granularity_val: str = field(default="64", metadata={"help": "landmark granularity if the pooling method is lmk_fixed (enter any str integer value) or lmk_var (x_y_..._z) where x,y,...,z are integers"})
    full_contrastive: bool = field(default=False, metadata={"help": "Use Full contrastive loss"})
    