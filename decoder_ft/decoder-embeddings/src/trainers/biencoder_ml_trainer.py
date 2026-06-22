import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict, Tuple, Any, Union, TYPE_CHECKING
from transformers.trainer import (
    DataLoader,
    _is_peft_model
)
from datasets import load_dataset, DatasetDict, Dataset

from logger_config import logger
from metrics import accuracy, batch_mrr


from models import BiencoderModel, BiencoderOutput
from trainers import BiencoderTrainer
from loaders import RetrievalDataLoader3
from collators import BiencoderCollator
import torch.distributed as dist
import pandas as pd
from numpy.random import multinomial

import math
import os
import shutil
import sys
import time
import inspect
import gc
import psutil
# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    hp_params,
)

# isort: on
import deepspeed
import torch.distributed as dist
from packaging import version
from torch.utils.data import DataLoader, RandomSampler

from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, deepspeed_optim_sched, is_deepspeed_available
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.trainer_callback import (
    TrainerState,
)
from transformers.trainer_pt_utils import (
    get_dataloader_sampler,
    get_model_param_count,
)
from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    has_length,
    seed_worker,
    speed_metrics,
)
from transformers.training_args import ParallelMode
from transformers.utils import (
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_safetensors_available,
    is_sagemaker_mp_enabled,
    logging,
)

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

# if is_torch_tpu_available(check_device=False):
#     import torch_xla.core.xla_model as xm
#     import torch_xla.debug.metrics as met
def is_torch_tpu_available(**kwargs):
    return False

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils.dataclasses import DistributedType
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler, _PYTORCH_DATALOADER_KWARGS, SkipDataLoader, DataLoaderShard

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if TYPE_CHECKING:
    import optuna


logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

def _unpack_qp(inputs: Dict[str, torch.Tensor], q_prefix='q_', d_prefix='d_', r_prefix='r_') -> Tuple:
    kd_labels_key = 'kd_labels'
    query_batch_dict = {k[len(q_prefix):]: v for k, v in inputs.items() if k.startswith(q_prefix)}
    doc_batch_dict = {k[len(d_prefix):]: v for k, v in inputs.items() if k.startswith(d_prefix)}
    reranker_batch_dict = {k[len(r_prefix):]: v for k, v in inputs.items() if k.startswith(r_prefix)}

    # if kd_labels_key in inputs:
    #     assert len(query_batch_dict) > 0
    #     query_batch_dict[kd_labels_key] = inputs[kd_labels_key]

    if not query_batch_dict:
        query_batch_dict = None
    if not doc_batch_dict:
        doc_batch_dict = None
    if kd_labels_key in inputs:
        return query_batch_dict, doc_batch_dict, {kd_labels_key: inputs[kd_labels_key]}, reranker_batch_dict
    return query_batch_dict, doc_batch_dict, None, reranker_batch_dict

def deepspeed_init_kd(trainer, num_training_steps, inference=False):
    # in innitialization, reduce the numper of training parameters in teacher
    # to avoid OOM error. The teacher will be set to eval model later anyway. 
    """
    Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.

    If `resume_from_checkpoint` was passed then an attempt to resume from a previously saved checkpoint will be made.

    Args:
        trainer: Trainer object
        num_training_steps: per single gpu
        resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load
        inference: launch in inference mode (no optimizer and no lr scheduler)
        auto_find_batch_size: whether to ignore the `train_micro_batch_size_per_gpu` argument as it's being
            set automatically by the auto batch size finder

    Returns: optimizer, lr_scheduler

    We may use `deepspeed_init` more than once during the life of Trainer, when we do - it's a temp hack based on:
    https://github.com/microsoft/DeepSpeed/issues/1394#issuecomment-937405374 until Deepspeed fixes a bug where it
    can't resume from a checkpoint after it did some stepping https://github.com/microsoft/DeepSpeed/issues/1612

    """
    from deepspeed.utils import logger as ds_logger

    model = trainer.model 
    args = trainer.args

    hf_deepspeed_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config

    # resume config update - some bits like `model` and `num_training_steps` only become available during train
    hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)

    # set the Deepspeed log level consistent with the Trainer
    ds_logger.setLevel(args.get_process_log_level())

    if inference:
        # only Z3 makes sense for the inference
        if not hf_deepspeed_config.is_zero3():
            raise ValueError("ZeRO inference only makes sense with ZeRO Stage 3 - please adjust your config")

        # in case the training config is re-used for inference
        hf_deepspeed_config.del_config_sub_tree("optimizer")
        hf_deepspeed_config.del_config_sub_tree("lr_scheduler")
        optimizer, lr_scheduler = None, None
        model_parameters = None
    else:
        trainer.optimizer = None  # important for when deepspeed_init is used as re-init
        for n, p in model.named_parameters():
            if 'teacher_model' in n:
                p.requires_grad = False
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer, lr_scheduler = deepspeed_optim_sched(
            trainer, hf_deepspeed_config, args, num_training_steps, model_parameters
        )

    # keep for quick debug:
    # from pprint import pprint; pprint(config)

    return optimizer, lr_scheduler

class MultiLoaderIterator:
    def __init__(self, output_dir, loaders, names, probs):
        self._loaders = loaders
        self._names = names
        self._probs = probs
        self._iters = [l.__iter__() for l in self._loaders]
        self.dataset = self._loaders[0].dataset  # trainer expects this member
        self._output_dir = output_dir
        self._count=0
        self._local_epoch = [0] * len(self._iters)
        self.skip_batches = 0
        self.batch_index = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        choice = multinomial(1, self._probs).argmax()

        while True:
            self.batch_index += 1
            try:
                batch=next(self._iters[choice])
            except StopIteration as x:
                ep=self._local_epoch[choice]+1
                with open(self._output_dir + '/dataloader.'+str(os.getpid()), 'at') as pidOut:
                    print('reshuffling: ', self._names[choice], 'local epoch: ', ep, file=pidOut)
                # believe it or not, set_epoch reshuffles ...
                self._loaders[choice].dataset.set_epoch(ep)
                # calling __iter__() resets it
                self._iters[choice]=self._loaders[choice].__iter__()
                batch=next(self._iters[choice])
                self._local_epoch[choice] = ep

            # YL: removed to work with transformer 4.37 dataloader
            # if self._count<10:
            #     with open(self._output_dir + '/dataloader.'+str(os.getpid()), 'at') as pidOut:
            #         print(self._count, ' pid: ',os.getpid(), 'choosing: ', self._names[choice], file=pidOut)
            #         print('query_ids:', batch['q_query_ids'], file=pidOut)
            #     self._count = self._count+1
            if self.batch_index > self.skip_batches:
                # print('Returning batch')
                return batch
    
class MultiLoaderIteratorEval(MultiLoaderIterator):
    def __init__(self, output_dir, loaders, names, probs):
        super().__init__(output_dir, loaders, names, probs)

    def __next__(self):
        choice = multinomial(1, self._probs).argmax()

        try:
            batch=next(self._iters[choice])
            return batch
        except:
            raise StopIteration






class BiencoderMLTrainer(BiencoderTrainer):
    def __init__(self, dataset_config_handler : RetrievalDataLoader3, *pargs, **kwargs):
        super(BiencoderMLTrainer, self).__init__(*pargs, **kwargs)
        self.rdl3 = dataset_config_handler
        self.model: BiencoderModel
        # self.accelerator.dispatch_batches = False # to work with accelerator library, True will cause error in multi-GPU case

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
        datasets=self.rdl3.base_datasets
        names=self.rdl3.base_dataset_names
        probs=self.rdl3.base_dataset_probs

        self._loaders = []
        num_workers_tmp = self.args.dataloader_num_workers  # UGLY
        self.args.dataloader_num_workers=min(1,num_workers_tmp)
        for i, (dc, name) in enumerate(zip(datasets,names)):
            self.train_dataset=self.rdl3.base_datasets[i]
            dl = super().get_train_dataloader()
            self.train_dataset=None
            self._loaders.append(dl)
        self.args.dataloader_num_workers=num_workers_tmp


        mdl = MultiLoaderIterator(self.args.output_dir, self._loaders, names, probs)
        return mdl 


    def compute_loss(self, model, inputs, teacher_model=None, return_outputs=False,num_items_in_batch=None):
        query, passage, kd_labels, reranker_inps = _unpack_qp(inputs)
        if self.args.teacher:
            query_teacher, passage_teacher, _, _ = _unpack_qp(inputs, q_prefix='teacher_q_', d_prefix='teacher_d_')
            outputs: BiencoderOutput = model(query=query, passage=passage, query_teacher=query_teacher, passage_teacher=passage_teacher)
        else:
            outputs: BiencoderOutput = model(query=query, passage=passage, kd_labels=kd_labels, reranker_inps=reranker_inps)
        loss = outputs.loss

        if self.model.training:
            step_acc1, step_acc3, step_acc10, step_acc50 = accuracy(output=outputs.scores.detach(), target=outputs.labels, topk=(1, 3, 10, 50))
            step_mrr = batch_mrr(output=outputs.scores.detach(), target=outputs.labels)

            self.acc1_meter.update(step_acc1)
            self.acc3_meter.update(step_acc3)
            self.acc10_meter.update(step_acc10)
            self.acc50_meter.update(step_acc50)
            self.mrr_meter.update(step_mrr)

            if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                log_info = ', '.join(map(str, [self.mrr_meter, self.acc1_meter, self.acc3_meter, self.acc10_meter, self.acc50_meter]))
                logger.info('step: {}, {}'.format(self.state.global_step, log_info))

            self._reset_meters_if_needed()

        return (loss, outputs) if return_outputs else loss

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        # overload to use custmized deepspeed init function
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            if self.args.teacher:
                self.optimizer, self.lr_scheduler = deepspeed_init_kd(self, num_training_steps=max_steps)
            else:
                self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            print('Using accelerator prepare')
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not self.args.save_and_reload_student_only and not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        start_time = time.time()
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                # skip batches are handled directly in MultiLoaderIterator, because that is where
                # reshuffle happens when dataset reach to the end
                epoch_iterator.skip_batches = steps_trained_in_current_epoch
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            # rank = dist.get_rank()
            # print('Epoch Rank', rank)
            epoch_iterator = iter(epoch_iterator)
            # for step, inputs in enumerate(epoch_iterator):
            while True:
                try:
                    # if rank==0:
                    #     print('before iterator' + str(os.system("free -h")) + str(psutil.Process().pid))
                    inputs = next(epoch_iterator)
                    # if rank==0:
                    #     print('after iterator' + str(os.system("free -h")) + str(psutil.Process().pid))
                except StopIteration:
                    break
                total_batched_samples += 1
                # print('\n'*10)
                # if rank==0:
                #     print('Start' + str(os.system("free -h")) + str(psutil.Process().pid))
                # gc.collect()
                # torch.cuda.empty_cache()
                # process = psutil.Process()
                # mem_info = process.memory_info()
                # # print("Total batched samples", total_batched_samples)
                # print(f"Current PID: {psutil.Process().pid}", f"RSS: {mem_info.rss/1e6:.2f} MB, VMS: {mem_info.vms/1e6:.2f} MB")

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping
                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                        else:
                            grad_norm = _grad_norm.item() if _grad_norm is not None else None


                    # Optimizer step
                    # if dist.get_rank()==0:
                    #     print('Before step')
                    #     for k, v in self.optimizer.state.items():
                    #         print(k.shape, v)
                    #     for i, pg in enumerate(self.optimizer.param_groups):
                    #         print(i, "lr:", pg['lr'], "num params:", len(pg['params']))
                    self.optimizer.step()
                    # if dist.get_rank()==0:
                    #     print('After step')
                    #     for n,p in model.named_parameters():
                    #         print(n, p.sum())
                    #         # full_grad = deepspeed.utils.safe_get_full_grad(p)
                    #         full_grad = self.optimizer.state[p]
                    #         if full_grad != None:
                    #             print(n, full_grad.keys())
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_tpu_available():
                        xm.mark_step()
                    break
                step += 1
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _save_optimizer_and_scheduler(self, output_dir):
        # overload to save always only save trainable parameters in distillation
        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
        elif self.is_deepspeed_enabled:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            if self.args.save_and_reload_student_only:
                self.model_wrapped.save_checkpoint(output_dir, exclude_frozen_parameters=True)
            else:
                accept_exclude_frozen_parameters = "exclude_frozen_parameters" in set(
                    inspect.signature(self.model_wrapped.save_checkpoint).parameters.keys()
                )
                if accept_exclude_frozen_parameters and _is_peft_model(self.model):
                    self.model_wrapped.save_checkpoint(output_dir, exclude_frozen_parameters=True)
                else:
                    self.model_wrapped.save_checkpoint(output_dir)
        elif self.is_fsdp_enabled:
            # save fsdp specific ckpt for resuming from ckpt
            save_fsdp_model(
                self.accelerator.state.fsdp_plugin, self.accelerator, self.model, output_dir, **_get_fsdp_ckpt_kwargs()
            )
            save_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin, self.accelerator, self.optimizer, self.model, output_dir
            )
        elif self.args.should_save:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

        # Save SCHEDULER & SCALER
        is_deepspeed_custom_scheduler = self.is_deepspeed_enabled and not isinstance(
            self.lr_scheduler, DeepSpeedSchedulerWrapper
        )
        if (
            self.args.should_save
            and (not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler)
            and not is_torch_tpu_available()
        ):
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)