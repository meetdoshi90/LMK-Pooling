#!/bin/bash

SCRIPT="decoder_ruler_eval.py"

BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"

COMMON_ARGS="--peft --corpus_size 100 --num_examples 100"

LMK_PATH="/dccstor/sdg/meet/models/decoder_embedding/lmk_exps/bge_en_icl_data_kd/Mistral-7B-Instruct-v0.3/loss_false_layers__scale__nnodes_1_gpus_8_temp_0.02_lr_2e-4_wd_3e-2_pool_lmk_gran_32,64,128,256"

LAST_PATH="/dccstor/sdg/meet/models/decoder_embedding/lmk_exps/bge_en_icl_data_kd/Mistral-7B-Instruct-v0.3/loss_false_layers__scale__nnodes_1_gpus_8_temp_0.02_lr_2e-4_wd_3e-2_pool_last_lmkexps"

MEAN_PATH="/dccstor/sdg/meet/models/decoder_embedding/lmk_exps/bge_en_icl_data_kd/Mistral-7B-Instruct-v0.3/loss_false_layers__scale__nnodes_1_gpus_8_temp_0.02_lr_2e-4_wd_3e-2_pool_mean_lmkexps"

CHECKPOINTS=(500 1000)

for CKPT in "${CHECKPOINTS[@]}"; do

    bsub \
    -J lmk_${CKPT} \
    -n 1 \
    -R "span[ptile=1]" \
    -R "rusage[mem=400G]" \
    -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH10080GBHBM3" \
    "python3 ${SCRIPT} \
    --base_model ${BASE_MODEL} \
    --model ${LMK_PATH}/checkpoint-${CKPT}/ \
    ${COMMON_ARGS} \
    --pool lmk \
    --lmk_granularity 2048"

    # bsub \
    # -J last_${CKPT} \
    # -n 1 \
    # -R "span[ptile=1]" \
    # -R "rusage[mem=400G]" \
    # -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH10080GBHBM3" \
    # "python3 ${SCRIPT} \
    # --base_model ${BASE_MODEL} \
    # --model ${LAST_PATH}/checkpoint-${CKPT}/ \
    # ${COMMON_ARGS} \
    # --pool last"

    # bsub \
    # -J mean_${CKPT} \
    # -n 1 \
    # -R "span[ptile=1]" \
    # -R "rusage[mem=400G]" \
    # -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH10080GBHBM3" \
    # "python3 ${SCRIPT} \
    # --base_model ${BASE_MODEL} \
    # --model ${MEAN_PATH}/checkpoint-${CKPT}/ \
    # ${COMMON_ARGS} \
    # --pool mean"

done