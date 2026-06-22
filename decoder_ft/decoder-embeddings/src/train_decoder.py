import logging
import os

import torch
from typing import Dict
from functools import partial
from transformers.utils.logging import enable_explicit_format
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_utils import get_last_checkpoint

from transformers import (
    AutoTokenizer,
    AutoModel,
    Mxfp4Config,
    HfArgumentParser,
    EvalPrediction,
    Trainer,
    set_seed,
    PreTrainedTokenizerFast
)

from logger_config import logger, LoggerCallback
from config import Arguments
from trainers import BiencoderMLTrainer
from loaders import RetrievalDecoderDataLoader
from collators import DecoderCollator
from metrics import accuracy, batch_mrr
from models import DecoderModel
import subprocess
from peft import LoraConfig, TaskType, get_peft_model


def get_nccl_socket_ifname():
    ipa = subprocess.run(['ip', 'a'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = ipa.stdout.decode('utf-8').split('\n')
    all_names = []
    name = None
    for line in lines:
        if line and not line[0] == ' ':
            name = line.split(':')[1].strip()
            continue
        if 'link/infiniband' in line:
            all_names.append(name)
    os.environ['NCCL_SOCKET_IFNAME'] = ','.join(all_names)


def fix_infiniband():
    # os.environ['NCCL_SOCKET_IFNAME'] = "^lo,docker,virbr,vmnet,vboxnet,wl,ww,ppp,bond"

    # ifname = os.environ.get('NCCL_SOCKET_IFNAME', None)
    # if ifname is None:
    #     os.environ['NCCL_SOCKET_IFNAME'] = '^lo,docker0'
    get_nccl_socket_ifname()
    os.environ['NCCL_IB_CUDA_SUPPORT'] = '1'
    ibv = subprocess.run('ibv_devinfo', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = ibv.stdout.decode('utf-8').split('\n')
    exclude = ''
    include = ''
    for line in lines:
        if 'hca_id:' in line:
            name = line.split(':')[1].strip()
        if '\tport:' in line:
            port = line.split(':')[1].strip()
        if 'link_layer:' in line and 'Ethernet' in line:
            exclude = exclude + f'{name}:{port},'
        if 'link_layer:' in line and 'infiniband' in line.lower():
            include = include + f'{name}:{port},'
    if exclude:
        exclude = '^' + exclude[:-1]
        # print(exclude)
        os.environ['NCCL_IB_HCA'] = exclude
    else:
        os.environ['NCCL_IB_HCA'] = include[:-1]

def _common_setup(args: Arguments):
    if args.process_index > 0:
        logger.setLevel(logging.WARNING)
    enable_explicit_format()
    set_seed(args.seed)

    # Detecting last checkpoint.
    last_checkpoint = None

    if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint



def _compute_metrics(args: Arguments, eval_pred: EvalPrediction) -> Dict[str, float]:
    # field consistent with BiencoderOutput
    preds = eval_pred.predictions
    scores = torch.tensor(preds[-1]).float()
    labels = torch.arange(0, scores.shape[0], dtype=torch.long) * args.train_n_passages
    labels = labels % scores.shape[1]

    topk_metrics = accuracy(output=scores, target=labels, topk=(1, 3))
    mrr = batch_mrr(output=scores, target=labels)

    return {'mrr': mrr, 'acc1': topk_metrics[0], 'acc3': topk_metrics[1]}

def main():
    get_nccl_socket_ifname()
    fix_infiniband()
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    last_checkpoint = _common_setup(args)
    logger.info('Args={}'.format(str(args)))


    model: DecoderModel = DecoderModel(args=args)
    try:
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_name_or_path)
    except Exception as e:
        logger.error(f'Tokenizer not found in model path; loading mistral v3 tokenizer as default')
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

    logger.info(f'Padding side default: {tokenizer.padding_side}')
    tokenizer.padding_side = "left"
    pad_id = model.model.config.pad_token_id
    logger.info(f'Pad id: {pad_id}')
    logger.info(f'Bos id: {model.model.config.bos_token_id}')
    logger.info(f'Eos id: {model.model.config.eos_token_id}')
    if pad_id is not None:
        pad_token_str = tokenizer.convert_ids_to_tokens(pad_id)
        logger.info(f'Setting tokenizer pad token as {pad_token_str}')
        tokenizer.pad_token = pad_token_str
        tokenizer.pad_token_id = pad_id
    if tokenizer.pad_token is None:
        logger.info(f'Setting tokenizer pad token as {tokenizer.unk_token}')
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    
    if args.use_reranker_for_decoder_distillation != None:
        reranker_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.use_reranker_for_decoder_distillation)
        reranker_tokenizer.padding_side = "right"
    else:
        reranker_tokenizer = None

    if args.pooling_source == 'lmk':
        # EOS is the landmark token — no vocab changes needed
        args.lmk_token_id = tokenizer.eos_token_id
        logger.info(f"LMK pooling: landmark token = EOS (id={args.lmk_token_id}), "
                    f"granularity={args.lmk_granularity}, "
                    f"granularity_set={getattr(args, 'lmk_granularity_set', None)}")
    
    #peft, for mistral models
    # target_modules=['q_proj',
    #                 'k_proj',
    #                 'v_proj',
    #                 'o_proj',
    #                 'gate_proj',
    #                 'up_proj',
    #                 'down_proj',
    #                 'lm_head',
    #                 'embed_tokens']
    # Following bge-en-icl
    if 'gpt-oss' in args.model_name_or_path:
        param_names = [
            name for name, _ in model.model.named_parameters()
            if "mlp.experts" in name and (
                name.startswith("model.layers.") and
                ("gate_up_proj" in name or "down_proj" in name)
            )
        ]
        logger.info(f'Router lora param target names: {param_names}')
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            target_modules=['q_proj',
                        'k_proj',
                        'v_proj',
                        'o_proj',
                        'gate_proj',
                        'up_proj',
                        'down_proj',
                        'in_proj',],
            target_parameters=param_names,
        )
        model.model.enable_input_require_grads()
        model.model = get_peft_model(model.model, peft_config)
        model.model.print_trainable_parameters()

    else:
        if args.freeze_layers_upto != None:
            target_modules = []
            for name, module in model.named_modules():
                if "model.layers." in name:
                    layer_idx = int(name.split('.')[3])
                    if layer_idx >= args.freeze_layers_upto:
                        for proj in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','in_proj']:
                            if name.endswith(proj):
                                print(name)
                                target_modules.append(name[len('model.'):])  # or module name pattern
            print(target_modules)
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=32, lora_dropout=0.1, target_modules=target_modules)
        else:
            target_modules=['q_proj',
                            'k_proj',
                            'v_proj',
                            'o_proj',
                            'gate_proj',
                            'up_proj',
                            'down_proj',
                            'in_proj',] # no out_proj since its incompatible in peft for mamba models
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=32, lora_dropout=0.1, target_modules=target_modules)
        model.model.enable_input_require_grads()
        model.model = get_peft_model(model.model, peft_config)
        model.model.print_trainable_parameters()

    print(dir(model.model))
    print(dir(model.model.model))
    for n,p in model.model.named_parameters():
        print(n, p.shape, p.requires_grad)
    print(model.model.print_trainable_parameters())
    print(model.model.get_layer_status())
    print(model.model.get_model_status())

    logger.info(model)
    logger.info('Vocab size: {}'.format(len(tokenizer)))

    # if args.freeze_pos_emb:
    #     for p in list(model.embeddings.position_embeddings.parameters()):
    #         p.requires_grad=False

    data_collator = DecoderCollator(
        tokenizer=tokenizer,
        reranker_tokenizer=reranker_tokenizer,
        pad_to_multiple_of=8 if args.fp16 else None, args=args)
    retrieval_data_loader = RetrievalDecoderDataLoader(args=args, tokenizer=tokenizer, reranker_tokenizer=reranker_tokenizer)
    train_dataset = retrieval_data_loader.train_dataset
    eval_dataset = retrieval_data_loader.eval_dataset

    trainer: Trainer = BiencoderMLTrainer(
        dataset_config_handler=retrieval_data_loader,
        model=model,
        args=args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=partial(_compute_metrics, args),
        tokenizer=tokenizer,
        # reranker_tokenizer=reranker_tokenizer,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LoggerCallback)
    retrieval_data_loader.trainer = trainer
    model.trainer = trainer

    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        #save_sentence_encoder_config(model, args.output_dir, args)

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset) if hasattr(train_dataset, '__len__') else 0
                                                                 
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        metrics["eval_samples"] = len(eval_dataset) if hasattr(eval_dataset, '__len__') else 0

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return


if __name__ == "__main__":
    main()
