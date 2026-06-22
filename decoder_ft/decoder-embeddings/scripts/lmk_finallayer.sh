#!/bin/bash

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=error

MASTER_ADDRESS=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
HOST=$(echo $HOSTNAME | cut -d '.' -f1)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $HOST | cut -d':' -f1)-1))
MASTER_PORT=$((10000 + $RANDOM % 50000))

NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)
NNODE=$NNODES
NPROC_PER_NODE=$GPUS_PER_NODE

MULTI_LAYER_LOSS=false
# MULTI_LAYER_LOSS_LAYERS="2,4,8,16,32"
# MULTI_LAYER_LOSS_SCALE='layer_scale'

MULTI_LAYER_ARGS=()
[ "$MULTI_LAYER_LOSS" = "true" ] && MULTI_LAYER_ARGS+=(--multi_layer_loss)
[ -n "$MULTI_LAYER_LOSS_SCALE" ] && MULTI_LAYER_ARGS+=(--multi_layer_loss_scale "$MULTI_LAYER_LOSS_SCALE")
[ -n "$MULTI_LAYER_LOSS_LAYERS" ] && {
    MULTI_LAYER_ARGS+=(--multi_layer_loss_layers)
    for i in ${MULTI_LAYER_LOSS_LAYERS//,/ }; do
        MULTI_LAYER_ARGS+=("$i")
    done
}

echo "${MULTI_LAYER_ARGS[@]}"

LAYERS=$(echo $MULTI_LAYER_LOSS_LAYERS | tr ',' '-')
SCALE=$MULTI_LAYER_LOSS_SCALE

echo MASTER_ADDRESS ${MASTER_ADDRESS}
echo HOSTNAME ${HOSTNAME}
echo HOST ${HOST}
echo NODE_RANK ${NODE_RANK}
echo MASTER_PORT ${MASTER_PORT}

TRAIN_DATA_CONFIG=/dccstor/embedding/meet/DDS/decoder-embeddings-evaluation/decoder-ft-vignesh/simlm/bge_en_icl_config.json
VAL_DATA_CONFIG=/dccstor/embedding/meet/DDS/decoder-embeddings-evaluation/decoder-ft-vignesh/simlm/slate_stage1_hard_negative_val.json
DATASETS_DIR=/dccstor/irl-rag/meet/data/decoder_embedding/bge-en-icl/formatted_data/

# MODEL_NAME_OR_PATH=ibm-granite/granite-3.3-8b-instruct #mistralai/Mistral-7B-Instruct-v0.3
MODEL_NAME_OR_PATH=/proj/embedding/meet/Landmark/inf-dds/eval_scripts/llm2vec_llama_3_8b_inst_mntp_unsup_simcse
# MODEL_NAME=$(basename "${MODEL_NAME_OR_PATH}" | tr '/' '_')
MODEL_NAME=$(echo "${MODEL_NAME_OR_PATH}" | awk -F'/' '{print $(NF-1)"_"$NF}')

MAX_STEPS=2000
WARMUP_STEPS=50
NUM_PASSAGES=8
PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_CACHE_MINIBATCH_SIZE=8
DATALOADER_NUM_WORKERS=0

SEED=0
TEMP=0.02
LR=2e-4
WD=3e-2
Q_MAX_LEN=128
P_MAX_LEN=512
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=20
DS_CONFIG=/dccstor/embedding/meet/DDS/decoder-embeddings-evaluation/decoder-ft-vignesh/simlm/ds_config/bf16_kd.json
ACCEL_CONFIG=/dccstor/embedding/meet/DDS/decoder-embeddings-evaluation/decoder-ft-vignesh/simlm/accel_config.json

# ── CHANGED: pooling mode ──────────────────────────────────────────────────────
POOLING_SOURCE=lmk

# ── NEW: LMK granularity settings ─────────────────────────────────────────────
# Option A — fixed granularity (one LMK every N tokens)
LMK_GRANULARITY=64

# Option B — variable granularity (uncomment to enable; overrides LMK_GRANULARITY)
LMK_GRANULARITY_SET="32,64,128,256"
# ──────────────────────────────────────────────────────────────────────────────

# Build LMK args conditionally so the script works for both options
LMK_ARGS=(--lmk_granularity ${LMK_GRANULARITY})
[ -n "$LMK_GRANULARITY_SET" ] && LMK_ARGS+=(--lmk_granularity_set "${LMK_GRANULARITY_SET}")

OUTDIR_BASE="/dccstor/sdg/meet/models/decoder_embedding/lmk_exps/bge_en_icl_data_kd/${MODEL_NAME}"

# ── CHANGED: outdir now encodes lmk granularity so runs don't collide ─────────
LMK_TAG="gran_${LMK_GRANULARITY_SET:-${LMK_GRANULARITY}}"
OUTDIR="${OUTDIR_BASE}/loss_${MULTI_LAYER_LOSS}_layers_${LAYERS}_scale_${SCALE}_nnodes_${NNODES}_gpus_${GPUS_PER_NODE}_temp_${TEMP}_lr_${LR}_wd_${WD}_pool_${POOLING_SOURCE}_${LMK_TAG}/"
mkdir -p "${OUTDIR}"

cp $0 ${OUTDIR}
cp ${TRAIN_DATA_CONFIG} ${OUTDIR}
cp ${VAL_DATA_CONFIG} ${OUTDIR}
cp ${DS_CONFIG} ${OUTDIR}

echo ${OUTDIR}

python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} --nnode=${NNODE} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDRESS} --master_port ${MASTER_PORT} \
    src/train_decoder.py --deepspeed ${DS_CONFIG} --accelerator_config ${ACCEL_CONFIG} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 16 \
    --add_pooler False \
    --t ${TEMP} \
    --seed ${SEED} \
    --do_train --data_dir ${DATASETS_DIR}/ \
    --train_data_config ${TRAIN_DATA_CONFIG} \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 500 \
    --val_data_config ${VAL_DATA_CONFIG} \
    --bf16 \
    --q_max_len ${Q_MAX_LEN} --p_max_len ${P_MAX_LEN} --train_n_passages ${NUM_PASSAGES} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --num_train_epochs 1 \
    --learning_rate ${LR} \
    --weight_decay ${WD} \
    --use_scaled_loss True \
    --warmup_steps ${WARMUP_STEPS} \
    --share_encoder True \
    --logging_steps 5 \
    --logging_dir ${OUTDIR}/runs \
    --output_dir ${OUTDIR}/ \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS} \
    --remove_unused_columns False \
    --disable_tqdm False \
    --max_steps ${MAX_STEPS} \
    --freeze_pos_emb \
    --no_titles \
    --pooling_source ${POOLING_SOURCE} \
    --overwrite_output_dir \
    --gradient_checkpointing \
    --dataloader_persistent_workers False \
    --dataloader_pin_memory False \
    --dataloader_drop_last True \
    --do_kd_biencoder True \
    --kd_cont_loss_weight 1.0 \
    --kd_mask_hn False \
    --cache_minibatch_size ${PER_DEVICE_CACHE_MINIBATCH_SIZE} \
    --full_contrastive_loss False \
    "${MULTI_LAYER_ARGS[@]}" \
    "${LMK_ARGS[@]}"                    # ── NEW