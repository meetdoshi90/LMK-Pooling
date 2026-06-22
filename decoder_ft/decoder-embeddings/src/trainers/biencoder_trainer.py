import os
import torch

from typing import Optional, Dict, Tuple
from transformers.trainer import Trainer

from logger_config import logger
from metrics import accuracy, batch_mrr
from models import BiencoderOutput, BiencoderModel
from utils import AverageMeter
from utils import save_sentence_encoder_config
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer import TRAINER_STATE_NAME
import numpy as np


def _unpack_qp(inputs: Dict[str, torch.Tensor]) -> Tuple:
    q_prefix, d_prefix, kd_labels_key = 'q_', 'd_', 'kd_labels'
    query_batch_dict = {k[len(q_prefix):]: v for k, v in inputs.items() if k.startswith(q_prefix)}
    doc_batch_dict = {k[len(d_prefix):]: v for k, v in inputs.items() if k.startswith(d_prefix)}

    if kd_labels_key in inputs:
        assert len(query_batch_dict) > 0
        query_batch_dict[kd_labels_key] = inputs[kd_labels_key]

    if not query_batch_dict:
        query_batch_dict = None
    if not doc_batch_dict:
        doc_batch_dict = None

    return query_batch_dict, doc_batch_dict


class BiencoderTrainer(Trainer):
    def __init__(self, *pargs, **kwargs):
        super(BiencoderTrainer, self).__init__(*pargs, **kwargs)
        self.model: BiencoderModel

        self.acc1_meter = AverageMeter('Acc@1', round_digits=2)
        self.acc3_meter = AverageMeter('Acc@3', round_digits=2)
        self.acc10_meter = AverageMeter('Acc@10', round_digits=2)
        self.acc50_meter = AverageMeter('Acc@50', round_digits=2)
        self.mrr_meter = AverageMeter('mrr', round_digits=2)
        self.last_epoch = 0
        self.current_gradient_accumulation_steps = self.args.gradient_accumulation_steps

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to {}".format(output_dir))
        self.model.save(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        save_sentence_encoder_config(self.model, output_dir, self.args)

    def _save_checkpoint(self, model, trial, metrics=None):
        # Overloaded from HF trainer.py, avoid error in multinode case when running on CCC
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            logger.warning(
                f"Checkpoint destination directory {output_dir} already exists and is non-empty. "
                "Saving will proceed but saved results may be invalid."
            )
            staging_output_dir = output_dir
        else:
            staging_output_dir = os.path.join(run_dir, f"tmp-{checkpoint_folder}")
        print(f'staging_output_dir: {staging_output_dir}')
        # here it saves the student part as separate model, for later evaluation 
        # save only from the main process
        if self.args.should_save:
            self._save(output_dir=staging_output_dir)

        if not self.args.save_only_model:
            # here it save the state_dict, optimizer, schedule all together, for checkpoint reloading
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(staging_output_dir)
            # Save RNG state
            self._save_rng_state(staging_output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(staging_output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(staging_output_dir)

        # Place checkpoint in final location after all saving is finished.
        # First wait for everyone to finish writing
        self.args.distributed_state.wait_for_everyone()

        # Then go through the rewriting process, only renaming and rotating from main process(es)
        if self.is_local_process_zero() if self.args.save_on_each_node else self.is_world_process_zero():
            if staging_output_dir != output_dir:
                if os.path.exists(staging_output_dir):
                    os.rename(staging_output_dir, output_dir)

                    # Ensure rename completed in cases where os.rename is not atomic
                    # And can only happen on non-windows based systems
                    if os.name != "nt":
                        fd = os.open(output_dir, os.O_RDONLY)
                        os.fsync(fd)
                        os.close(fd)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)
        # elif self.is_local_process_zero():
        #     # Clean up the remaining staging checkpoint folders on other nodes
        #     if staging_output_dir != output_dir and os.path.exists(staging_output_dir):
        #         shutil.rmtree(staging_output_dir)

        self.args.distributed_state.wait_for_everyone()

    def compute_loss(self, model, inputs, return_outputs=False):
        query, passage = _unpack_qp(inputs)
        outputs: BiencoderOutput = model(query=query, passage=passage)
        loss = outputs.loss

        if self.model.training:
            step_acc1, step_acc3 = accuracy(output=outputs.scores.detach(), target=outputs.labels, topk=(1, 3))
            step_mrr = batch_mrr(output=outputs.scores.detach(), target=outputs.labels)

            self.acc1_meter.update(step_acc1)
            self.acc3_meter.update(step_acc3)
            self.mrr_meter.update(step_mrr)

            if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                log_info = ', '.join(map(str, [self.mrr_meter, self.acc1_meter, self.acc3_meter]))
                logger.info('step: {}, {}'.format(self.state.global_step, log_info))

            self._reset_meters_if_needed()

        return (loss, outputs) if return_outputs else loss

    def _reset_meters_if_needed(self):
        if int(self.state.epoch) != self.last_epoch:
            self.last_epoch = int(self.state.epoch)
            self.acc1_meter.reset()
            self.acc3_meter.reset()
            self.acc10_meter.reset()
            self.acc50_meter.reset()
            self.mrr_meter.reset()
