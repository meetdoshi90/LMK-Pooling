#!/bin/bash

if [[ -n "$WANDB_API_KEY" ]]; then
    wandb login --relogin "$WANDB_API_KEY"
fi

nvidia-smi 

export WANDB_MODE="offline"
export WANDB_DISABLED=true
export NCCL_SOCKET_IFNAME="ib,bond"
export NCCL_IB_CUDA_SUPPORT=1
export CUBLAS_WORKSPACE_CONFIG=:16:8
#export CUBLAS_WORKSPACE_CONFIG=""
export CUDA_LAUNCH_BLOCKING=1 

NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)
MASTER_ADDRESS=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
HOST=$(echo $HOSTNAME | cut -d '.' -f1)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $HOST | cut -d':' -f1)-1))
MASTER_PORT=$((10000 + $RANDOM % 50000))

PREFIX=$1
MODEL_NAME=$2
NNEG=$3
POOLING_METHOD=$4
LMK_GRANULARITY=$5

echo $PREFIX "-" $MODEL_NAME "-" $NNEG "-" $POOLING_METHOD "-" $LMK_GRANULARITY

if [ -z "$LMK_GRANULARITY" ]; then
    LMK_GRANULARITY="empty"
fi

TOKENIZERS_PARALLELISM=false \
torchrun --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=101 \
    --rdzv_endpoint=$MASTER_ADDRESS:$MASTER_PORT \
    -m FlagEmbedding.baai_general_embedding.finetune.run_20gpus_mdl \
    --output_dir ${OUTPUT_DIR}/multilingual_distil_${PREFIX}_${POOLING_METHOD}_gran_${LMK_GRANULARITY}/ \
    --model_name_or_path ${MODEL_NAME} \
    --train_data ${DATA_DIR}/bge-m3/bge-m3-langwise/ \
    --train_data_config data_configs/bge_config_weight.json \
    --learning_rate 2e-5 \
    --bf16 \
    --deepspeed config/ds_config.json \
    --accelerator_config config/accel_config.json \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.02 \
    --warmup_steps 250 \
    --query_max_len 128 \
    --passage_max_len 512 \
    --train_group_size ${NNEG} \
    --negatives_cross_device \
    --logging_steps 10 \
    --sentence_pooling_method ${POOLING_METHOD} \
    --landmark_granularity_val ${LMK_GRANULARITY} \
    --save_steps 1000 \
    --dataloader_num_workers 8 \
    --max_steps 10000 \
    --remove_unused_columns False \
    --fix_position_embedding False \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs='{"use_reentrant":"False"}' \
    --report_to none \
    --train_datasets_sampling_init "weight" \
    --is_train_data_streaming False \
    --is_neg_distil_collator True \
    --load_best_model_at_end False