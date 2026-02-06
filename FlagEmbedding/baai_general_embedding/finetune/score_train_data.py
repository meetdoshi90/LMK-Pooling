import logging
import os
from pathlib import Path
import torch

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from tqdm import tqdm
import json
from .arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from .data import TrainDatasetForEmbedding, ScoringCollator,EvalDatasetForEmbedding
from .modeling import BiEncoderModel
from .trainer import BiTrainer
from datasets import Dataset


logger = logging.getLogger(__name__)



def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    logger.info('Config: %s', config)
    model = BiEncoderModel(model_name=model_args.model_name_or_path,
                           normlized=model_args.normlized,
                           sentence_pooling_method=training_args.sentence_pooling_method,
                           negatives_cross_device=training_args.negatives_cross_device,
                           temperature=training_args.temperature)
    
    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    val_dataset = EvalDatasetForEmbedding(args=data_args, tokenizer=tokenizer)
    
    dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=training_args.per_device_eval_batch_size,collate_fn=ScoringCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len
        ))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    scores = []
    queries = []
    pos_list = []
    
    for batch in tqdm(dataloader):
        with torch.no_grad():
            output = model(batch["query"].to(device),batch["passage"].to(device))
        queries+=batch["q_orig"]
        pos_list+=batch["p_orig"]
        ss = torch.diagonal(output["scores"]).tolist()
        scores+= ss
    dataset_dict = {"query":queries,"pos":pos_list,"score":scores}
    ds = Dataset.from_dict(dataset_dict)
    # res = {i: scores[i] for i in range(len(scores))}
    # json.dump(res,open("scores_nq_fever_with_hn.json","w"))
    folder = "/dccstor/embedding/vk/data/271m_data/unsup/"
    out_path=folder+data_args.eval_data.split("/")[-1]+"_with_scores.hf"
    ds.save_to_disk(out_path)
if __name__=="__main__":
    main()