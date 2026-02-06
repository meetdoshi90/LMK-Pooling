export MASTER_ADDR=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
export MASTER_PORT=7501
export NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $HOSTNAME | cut -d':' -f1)-1))
export NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
export GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)
export WORLD_SIZE=$(($NNODES * $GPUS_PER_NODE))

model_name_or_path=$1
lr=$2
epochs=$3
bsz=$4
max_seq_length=$5
output_dir=$6
encoder_mlm_probability=$7
decoder_mlm_probability=$8


torchrun --nproc_per_node ${GPUS_PER_NODE}\
    -m FlagEmbedding.baai_general_embedding.retromae_pretrain.run \
    --output_dir ${output_dir} \
    --model_name_or_path ${model_name_or_path} \
    --train_data ${DATA_DIR}/multilingual/fineweb_all_languages/ \
    --encoder_mlm_probability ${encoder_mlm_probability} \
    --decoder_mlm_probability ${decoder_mlm_probability} \
    --learning_rate ${lr} \
    --num_train_epochs ${epochs} \
    --per_device_train_batch_size ${bsz} \
    --dataloader_drop_last True \
    --dataloader_num_workers ${GPUS_PER_NODE} \
    --max_seq_length ${max_seq_length} \
    --logging_steps 500 \
    --save_total_limit 2 \
    --save_steps 10000 \
    --bf16 \
    --train_landmark_embeddings True \
    --use_chunked_landmarks True \
    --report_to none 