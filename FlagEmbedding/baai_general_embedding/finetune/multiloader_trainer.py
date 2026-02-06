import os
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.trainer import (
    Trainer,
    DataLoader
)
from transformers.trainer import *
from transformers import TrainerCallback, DefaultFlowCallback
from transformers.tokenization_utils_base import BatchEncoding

import logging
from itertools import tee


import gc
from .modeling import BiEncoderModel
import pandas as pd
from numpy.random import multinomial

logger = logging.getLogger(__name__)


def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls'):
    word_embedding_model = models.Transformer(ckpt_dir,
                                              model_args={'trust_remote_code': True},
                                              config_args={'trust_remote_code': True},
                                              tokenizer_args={'trust_remote_code': True},
                                            )
    if pooling_mode in ['cls', 'lasttoken', 'max', 'mean', 'mean_sqrt_len_tokens', 'weightedmean']:
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
    else:
        model = SentenceTransformer(modules=[word_embedding_model], device='cpu')
    model.save(ckpt_dir)


def batch_to_device(batch, rank):
    """Move tensors and BatchEncoding to cuda:{rank}."""
    if rank=='cpu':
        device = torch.device(f"{rank}")
    else:
        device = torch.device(f"cuda:{rank}")
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, BatchEncoding):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: batch_to_device(v, rank) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(v, rank) for v in batch)
    return batch


class MultiLoaderIterator:
    def __init__(self, output_dir, loaders, names, probs, psis, weights, datasets_num_instances, batch_size, is_train):
        self._loaders = loaders
        self._names = names
        self._probs = probs
        self._psis = psis 
        self._weights = weights
        self._avg_weights = [0.0 for _ in weights] 
        self._weight_updates = 0
        # self._iters = [l.__iter__() for l in self._loaders]
        self._iters = [None] * len(self._loaders)
        self.dataset = self._loaders[0].dataset  # trainer expects this member
        self._output_dir = output_dir
        self._count=0
        self._local_epoch = [0] * len(self._iters)
        self.datasets_num_instances = datasets_num_instances//batch_size
        self.eval_count = 0
        self.is_train = is_train
        self.base_seed=1235
    
    def get_iter(self, idx):
        if self._iters[idx] is None:
            self._iters[idx] = iter(self._loaders[idx])
        return self._iters[idx]
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.datasets_num_instances
    
    def __next__(self, key=None): 
        if key==None:
            choice = multinomial(1, self._probs).argmax()
        else:
            if key < len(self._probs):
                choice = key
            else:
                raise Exception(key, 'Key out of bound. Must be less than', len(self._probs))
        try:
            it = self.get_iter(choice)
            batch=next(it)
        except StopIteration as x:
            ep=self._local_epoch[choice]+1
            if hasattr(self._loaders[choice].dataset, 'set_epoch'): # For iterable datasets
                self._loaders[choice].dataset.set_epoch(ep)
            else: # For sharded datasets
                new_seed = self.base_seed + ep # Use the initial seed plus the epoch number
                self._loaders[choice].dataset = self._loaders[choice].dataset.shuffle(seed=new_seed)
            old_iter = self._iters[choice]
            if old_iter is not None:
                del old_iter
                self._iters[choice] = None
                gc.collect()
            self._iters[choice] = self._loaders[choice].__iter__()
            batch = next(self._iters[choice])
            self._local_epoch[choice] = ep
            
        if (not self.is_train) and key==None:
            self.eval_count+=1
        else:
            self.eval_count=0

        if self.eval_count>self.datasets_num_instances:
            self.eval_count = 0
            for c in range(len(self._iters)):
                old_iter = self._iters[c]
                if old_iter is not None:
                    del old_iter
                    self._iters[c] = None
                    gc.collect()
                self._iters[c]=self._loaders[c].__iter__()
            raise StopIteration
        return batch
        
    def custom_next(self, key=None): 
        if key==None:
            choice = multinomial(1, self._probs).argmax() 
        else:
            if key < len(self._probs):
                choice = key
            else:
                raise Exception(key, 'Key out of bound. Must be less than', len(self._probs))
        try:
            it = self.get_iter(choice)
            batch=next(it)
        except StopIteration as x:
            ep=self._local_epoch[choice]+1
            if hasattr(self._loaders[choice].dataset, 'set_epoch'): # For iterable datasets
                self._loaders[choice].dataset.set_epoch(ep)
            else: # For sharded datasets
                new_seed = self.base_seed + ep # Use the initial seed plus the epoch number
                self._loaders[choice].dataset = self._loaders[choice].dataset.shuffle(seed=new_seed)
            old_iter = self._iters[choice]
            if old_iter is not None:
                del old_iter
                self._iters[choice] = None
                gc.collect()
            self._iters[choice] = self._loaders[choice].__iter__()
            batch = next(self._iters[choice])
            self._local_epoch[choice] = ep
            
        if (not self.is_train) and key==None:
            self.eval_count+=1
        else:
            self.eval_count=0

        if self.eval_count>=self.datasets_num_instances:
            print('Max eval steps reached')
            self.eval_count = 0
            raise StopIteration
        return batch



class AverageEvalLossCallback(TrainerCallback):
    def __init__(self):
        self.eval_losses = {}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        dataset_name = next((key for key in metrics if key.startswith('eval_') and key.endswith('_loss')), None)
        if dataset_name:
            self.eval_losses[dataset_name] = metrics[f'{dataset_name}']

        if len(self.eval_losses) == args.val_datasets_count:
            avg_eval_loss = sum(self.eval_losses.values()) / len(self.eval_losses)
            metrics['eval_loss'] = avg_eval_loss
            print(f"Average Evaluation Loss: {avg_eval_loss:.4f}")
            self.eval_losses.clear()



class BiencoderMLTrainer(Trainer):
    def __init__(self, dataset_handler, eval_dataset_handler, data_args=None, *pargs, **kwargs):
        super(BiencoderMLTrainer, self).__init__(*pargs, **kwargs)
        self.handler = dataset_handler
        self.eval_handler = eval_dataset_handler
        self.model: BiEncoderModel
        self.args.train_datasets_count = len(self.handler.train_datasets) 
        self.args.val_datasets_count = len(self.eval_handler.eval_datasets) if self.eval_handler.eval_datasets != None else 0
        self.data_args=data_args

        
    def get_train_dataloader(self) -> DataLoader:
        ''' create a dataloader that merges underlying dataloaders batch-by-batch
            the intent is that each batch comes from a single dataset
            we create a single dataloader for each set, using super().get_train_dataloader()
            this has the side effect of also having a collator for each dataset
            we return a merging iterator that (multinomial)samples from its list of dataloader
            and returns the (already-collated) batches, without an additional collator
            We also depend upon the MultiLoaderIterator having the same sequence of random
            numbers in each process in order to make the multi-gpu batch homogenous
        '''
        datasets=self.handler.train_datasets
        names=self.handler.train_datasets_names
        probs=self.handler.train_datasets_probs 
        psis=self.handler.train_datasets_psis
        weights=self.handler.train_datasets_weight
        datasets_num_instances=self.handler.train_datasets_num_instances
        self._loaders = []
        num_workers_tmp = self.args.dataloader_num_workers 
        self.args.dataloader_num_workers=1
        for i, (dc, name) in enumerate(zip(datasets,names)):
            self.train_dataset=self.handler.train_datasets[i]
            if self.data_args.is_train_data_streaming:
                dl = super().get_train_dataloader()
            else:
                base = super().get_train_dataloader()
                shard = base.dataset
                from torch.utils.data import RandomSampler
                sampler = RandomSampler(shard, replacement=True, num_samples=int(1e12))
                true_bs = base.batch_sampler.batch_size
                dl = base.__class__(
                    dataset=shard,
                    batch_size=true_bs,
                    sampler=sampler,
                    collate_fn=base.collate_fn,
                    drop_last=base.drop_last,
                    num_workers=base.num_workers,
                    pin_memory=base.pin_memory,
                    worker_init_fn=base.worker_init_fn,
                    persistent_workers=False,      # keep default for max throughput
                )
            self.train_dataset=None
            self._loaders.append(dl)
        self.args.dataloader_num_workers=num_workers_tmp
        mdl = MultiLoaderIterator(self.args.output_dir, self._loaders, names, probs, psis, weights, datasets_num_instances, self.args.train_batch_size, True)
        return mdl
    
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        ''' create a dataloader that merges underlying dataloaders batch-by-batch
            the intent is that each batch comes from a single dataset
            we create a single dataloader for each set, using super().get_train_dataloader()
            this has the side effect of also having a collator for each dataset
            we return a merging iterator that (multinomial)samples from its list of dataloader
            and returns the (already-collated) batches, without an additional collator
            We also depend upon the MultiLoaderIterator having the same sequence of random
            numbers in each process in order to make the multi-gpu batch homogenous
        '''
        #print('Eval dataset', eval_dataset)
        datasets=self.eval_handler.eval_datasets
        names=self.eval_handler.eval_datasets_names
        probs=self.eval_handler.eval_datasets_probs 
        psis=self.eval_handler.eval_datasets_psis
        weights=self.eval_handler.eval_datasets_weight
        datasets_num_instances=self.eval_handler.eval_datasets_num_instances
        self._loaders = []
        num_workers_tmp = self.args.dataloader_num_workers 
        self.args.dataloader_num_workers=1
        for i, (dc, name) in enumerate(zip(datasets,names)):
            self.train_dataset=self.eval_handler.eval_datasets[i]
            dl = super().get_eval_dataloader(self.train_dataset)
            self.train_dataset=None
            self._loaders.append(dl)
        self.args.dataloader_num_workers=num_workers_tmp
        mdl = MultiLoaderIterator(self.args.output_dir, self._loaders, names, probs, psis, weights, datasets_num_instances, self.args.eval_batch_size, False)
        return mdl

    def _save(self, output_dir: Optional[str] = None,state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir) 
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save the checkpoint for sentence-transformers library
        if self.is_world_process_zero():
            save_ckpt_for_sentence_transformers(output_dir, pooling_mode=self.args.sentence_pooling_method)

    
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False): #@meetdoshi ToDo: Update the compute_loss function for bi-level optimisation
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
    
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        if inputs is None:
            return (None, None, None)
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        labels = None

        with torch.no_grad():
            loss = None
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
            else:
                logits = outputs
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index - 1]
            loss = outputs.loss

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
    
    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint)
        model = self.model.load(best_model_path)
        return model
