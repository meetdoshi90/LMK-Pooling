#!/bin/bash

# BASE_DIR="/dccstor/sdg/meet/models/decoder_embedding/lmk_exps/bge_en_icl_data_kd/Mistral-7B-Instruct-v0.3/"
# BASE_DIR="/dccstor/sdg/meet/models/decoder_embedding/lmk_exps/msmarco_bge_en_icl_kd/Mistral-7B-Instruct-v0.3/"
# BASE_DIR="/dccstor/sdg/meet/models/decoder_embedding/lmk_exps/bge_en_icl_data_kd/granite-3.3-8b-instruct/"
# BASE_DIR="/dccstor/sdg/meet/models/decoder_embedding/lmk_exps/msmarco_bge_en_icl_kd/granite-4.1-8b/"
BASE_DIR="/dccstor/sdg/meet/models/decoder_embedding/lmk_exps/bge_en_icl_data_kd/eval_scripts_llm2vec_llama_3_8b_inst_mntp_unsup_simcse/"
CHECKPOINTS=(500 1000 1500 2000) #10000
EVAL_MULTILINGUAL=false
ADD_PROMPT=true   # always true
# OUTPUT_DIR="decoder_eval_results"
# OUTPUT_DIR="msmarco_decoder_eval_results"
# OUTPUT_DIR="granite_decoder_eval_results"
# OUTPUT_DIR="granite41_decoder_eval_results"
OUTPUT_DIR="llm2vec_llama38binst_decoder_eval_results"

# BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
# BASE_MODEL="ibm-granite/granite-3.3-8b-instruct"
# BASE_MODEL="ibm-granite/granite-4.1-8b"
BASE_MODEL="/proj/embedding/meet/Landmark/inf-dds/eval_scripts/llm2vec_llama_3_8b_inst_mntp_unsup_simcse"

MODELS=(
    loss_false_layers__scale__nnodes_1_gpus_8_temp_0.02_lr_2e-4_wd_3e-2_pool_lmk_gran_32,64,128,256
    loss_false_layers__scale__nnodes_1_gpus_8_temp_0.02_lr_2e-4_wd_3e-2_pool_mean_lmkexps
    loss_false_layers__scale__nnodes_1_gpus_8_temp_0.02_lr_2e-4_wd_3e-2_pool_last_lmkexps
)

TASK_NAMES=(
  LEMBNeedleRetrieval
  LEMBPasskeyRetrieval
  LEMBQMSumRetrieval
  LEMBSummScreenFDRetrieval
  LEMBWikimQARetrieval
  LEMBNarrativeQARetrieval
  ArguAna
  FiQA2018
  SCIDOCS
  CQADupstackUnixRetrieval
  CQADupstackGamingRetrieval
  ClimateFEVERHardNegatives
  FEVERHardNegatives
  HotpotQAHardNegatives
  TRECCOVID
  Touche2020Retrieval.v3
  AppsRetrieval
  CodeFeedbackMT
  CodeFeedbackST
  CodeTransOceanContest
  CodeTransOceanDL
  CosQA
  SyntheticText2SQL
  StackOverflowQA
  COIRCodeSearchNetRetrieval
  CodeSearchNetCCRetrieval
  MultiLongDocRetrieval
)


echo "Submitting jobs"

MLDR_LANGUAGES=(ara cmn deu eng fra hin ita jpn kor por rus spa tha)
MIRACL_LANGUAGES=(ara ben deu eng fas fin fra hin ind jpn kor rus spa swa tel tha yor zho)
MultiEURLEX_LANGUAGES=(hun por deu eng bul lit swe est spa nld pol fra slk fin mlt ita lav ces ell slv dan ron hrv)

for MODEL_NAME in "${MODELS[@]}"; do
    for TASK_NAME in "${TASK_NAMES[@]}"; do
        if [[ "$MODEL_NAME" == *"_last_"* ]]; then
            pool="last"
            split_sizes=(256)

        elif [[ "$MODEL_NAME" == *"_mean_"* ]]; then
            pool="mean"
            split_sizes=(256)

        elif [[ "$MODEL_NAME" == *"every_64"* ]]; then
            pool="mean_every_k"
            split_sizes=(256)

        elif [[ "$MODEL_NAME" == *"latent_attn"* ]]; then
            pool="latent_attn"
            split_sizes=(256)

        elif [[ "$MODEL_NAME" == *"lmk_en"* ]]; then
            pool="lmk"
            split_sizes=(256)

        elif [[ "$MODEL_NAME" == *"lmk_gran_32,64,128,256"* ]]; then
            pool="lmk"
            # split_sizes=(32 64 128 256)
            split_sizes=(32 64 128 256)

        elif [[ "$MODEL_NAME" == *"lmk_gran_32"* ]]; then
            pool="lmk"
            split_sizes=(32)

        elif [[ "$MODEL_NAME" == *"lmk_gran_64"* ]]; then
            pool="lmk"
            split_sizes=(64)

        elif [[ "$MODEL_NAME" == *"lmk_gran_128"* ]]; then
            pool="lmk"
            split_sizes=(128)

        elif [[ "$MODEL_NAME" == *"lmk_gran_256"* ]]; then
            pool="lmk"
            split_sizes=(256)

        else
            echo "⚠️ Unknown model type: $MODEL_NAME — skipping"
            continue
        fi

        MODEL_PATH="${BASE_DIR}/${MODEL_NAME}"
        LANGUAGES=("eng")  # default

        if $EVAL_MULTILINGUAL; then
            if [[ "$TASK_NAME" == *"MIRACL"* ]]; then
                LANGUAGES=("${MIRACL_LANGUAGES[@]}")
            elif [[ "$TASK_NAME" == *"MultiLongDocRetrieval"* ]]; then
                LANGUAGES=("${MLDR_LANGUAGES[@]}")
            elif [[ "$TASK_NAME" == *"MultiEURLEX"* ]]; then
                LANGUAGES=("${MultiEURLEX_LANGUAGES[@]}")
            fi
        fi

        for ckp in "${CHECKPOINTS[@]}"; do
            CKP_PATH="${MODEL_PATH}/checkpoint-${ckp}"

            if [[ ! -d "$CKP_PATH" ]]; then
                echo "❌ Missing checkpoint: $MODEL_NAME checkpoint-$ckp — skipping"
                # continue
                exit 0
            fi
            if [[ "$TASK_NAME" == LEMB* ]]; then
                MAX_SEQ_LEN=32768
            elif [[ "$TASK_NAME" == MultiEURLEX* ]]; then
                MAX_SEQ_LEN=8192
            else
                MAX_SEQ_LEN=8192
            fi
            
            for split_size in "${split_sizes[@]}"; do
                for lang in "${LANGUAGES[@]}"; do

                    MODEL_TAG=$(echo "$CKP_PATH" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

                    if [[ "$pool" == "lmk" ]]; then
                        POOL_TAG="${pool}_gran${split_size}"
                    else
                        POOL_TAG="${pool}"
                    fi

                    if $ADD_PROMPT; then
                        PROMPT_TAG="prompt"
                    else
                        PROMPT_TAG="noprompt"
                    fi

                    RUN_TAG="${lang}_${POOL_TAG}_${PROMPT_TAG}_${MAX_SEQ_LEN}_${MODEL_TAG}"

                    OUTPUT_PATH="${OUTPUT_DIR}/${TASK_NAME}_${RUN_TAG}"
                    if [[ -f "${OUTPUT_PATH}/${TASK_NAME}.json" ]]; then
                        echo "✅ Output path exists: ${OUTPUT_PATH}/${TASK_NAME}.json"
                    else
                        echo "❌ Output path does NOT exist: ${OUTPUT_PATH}/${TASK_NAME}.json"
                        echo "🚀 Submitting: $MODEL_NAME | ckp=$ckp | pool=$pool | split=$split_size | task=$TASK_NAME | lang=$lang | MSL=$MAX_SEQ_LEN"
                        bsub -q normal -n 1 \
                        -R "span[ptile=1]" \
                        -gpu "num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB" \
                        -M 400GB \
                        -J "${MODEL_NAME:0:20}-${ckp}-s${split_size}-t${TASK_NAME:0:15}" \
                        python decoder_landmark_index.py \
                            --base_model "$BASE_MODEL" \
                            --model "$CKP_PATH" \
                            --peft \
                            --max_seq_len $MAX_SEQ_LEN \
                            --pool "$pool" \
                            --lmk_granularity "$split_size" \
                            --task_type "$TASK_NAME" \
                            --lang "$lang" \
                            --output_dir "$OUTPUT_DIR" \
                            --add_prompt
                        sleep 0.1
                    fi
                done
            done
        done
    done
done
