import logging
import os
import sys
import subprocess

import transformers
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    AutoConfig,
    HfArgumentParser, set_seed, )
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)
from transformers.trainer_utils import is_main_process

from .arguments import DataTrainingArguments, ModelArguments
from .data import DatasetForPretraining, RetroMAECollator
from .modeling import RetroMAEForPretraining
from .trainer import PreTrainer

from pynvml import *

logger = logging.getLogger(__name__)


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

class GpuInfo(object):
    def __init__(self, index, memory_total, memory_used, gpu_load):
        """
        :param index: GPU index
        :param memory_total: total GPU memory, Mb
        :param memory_used: GPU memory already in use, Mb
        :param gpu_load: gpu utilization load, percents
        """
        self.index = int(index)
        self.memory_total = int(memory_total)
        self.memory_used = int(memory_used)
        try:
            self.gpu_load = int(gpu_load) / 100.
        except ValueError:
            # gpu utilization load is not supported in current driver
            self.gpu_load = 0.

    def __repr__(self):
        return "GPU #{}: memory total={} Mb, used={} Mb ({:.1f} %), gpu.load={}".format(
            self.index, self.memory_total, self.memory_used, 100. * self.memory_used / self.memory_total, self.gpu_load)

    def get_available_memory_portion(self):
        return (self.memory_total - self.memory_used) / self.memory_total


class NvidiaSmi(object):
    def __init__(self):
        command = "nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits".split()
        self.gpus = []
        try:
            process = subprocess.Popen(command,
                                       universal_newlines=True,
                                       stdout=subprocess.PIPE)
            stdout, stderr_ignored = process.communicate()
            for line in stdout.splitlines():
                index, memory_total, memory_used, gpu_load = line.split(', ')
                gpu = GpuInfo(index, memory_total, memory_used, gpu_load)
                self.gpus.append(gpu)
        except FileNotFoundError:
            # No GPU is detected. Try running `nvidia-smi` in a terminal."
            pass

    def get_gpus(self, min_free_memory=0., max_load=1.):
        """
        :param min_free_memory: filter GPUs with free memory no less than specified, between 0 and 1
        :param max_load: max gpu utilization load, between 0 and 1
        :return: list of available GpuInfo's
        """
        gpus = [gpu for gpu in self.gpus if gpu.get_available_memory_portion() >= min_free_memory and
                gpu.gpu_load <= max_load]
        return gpus

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
        

def set_cuda_visible_devices(limit_devices=int(1e9), min_free_memory=0.4, max_load=0.6) -> list:
    """
    Automatically sets CUDA_VISIBLE_DEVICES env to first `limit_devices` available GPUs with least used memory.
    :param limit_devices: limit available GPU devices to use
    :param min_free_memory: filter GPUs with free memory no less than specified, between 0 and 1
    :param max_load: max gpu utilization load, between 0 and 1
    """
    gpus = NvidiaSmi().get_gpus(min_free_memory, max_load)
    gpus.sort(key=lambda gpu: gpu.get_available_memory_portion(), reverse=True)
    limit_devices = min(limit_devices, len(gpus))
    gpus = gpus[:limit_devices]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu.index) for gpu in gpus)
    print("'CUDA_VISIBLE_DEVICES' is set to '{}'".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    return gpus
        
get_nccl_socket_ifname()
fix_infiniband()
#print(set_cuda_visible_devices())

class TrainerCallbackForSaving(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        control.should_save = True


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    training_args.remove_unused_columns = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    if training_args.local_rank in (0, -1):
        logger.info("Training/evaluation parameters %s", training_args)
        logger.info("Model parameters %s", model_args)
        logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    model_class = RetroMAEForPretraining
    collator_class = RetroMAECollator

    if model_args.model_name_or_path:
        model = model_class.from_pretrained(model_args, model_args.model_name_or_path)
        logger.info(f"------Load model from {model_args.model_name_or_path}------")
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    elif model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
        bert = BertForMaskedLM(config)
        model = model_class(bert, model_args)
        logger.info("------Init the model------")
        tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name)
    else:
        raise ValueError("You must provide the model_name_or_path or config_name")

    dataset = DatasetForPretraining(data_args.train_data)

    data_collator = collator_class(tokenizer,
                                   encoder_mlm_probability=data_args.encoder_mlm_probability,
                                   decoder_mlm_probability=data_args.decoder_mlm_probability,
                                   max_seq_length=data_args.max_seq_length)

    # Initialize our Trainer
    trainer = PreTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.add_callback(TrainerCallbackForSaving())

    # # Training
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    print_gpu_utilization()
    trainer.save_model()  # Saves the tokenizer too for easy upload


if __name__ == "__main__":
    main()