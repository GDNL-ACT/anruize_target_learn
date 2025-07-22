#!/bin/bash

# Set the NCCL timeout to 600000 milliseconds (10 minutes)
export NCCL_TIMEOUT=60000
export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

MODEL_NAME="mistral-7b"
BASE_MODEL_PATH="/nas-retrieval/zhijie.nzj/models/Mistral-7B-Instruct-v0.2"
DATA_PATH="/mnt/workspace/zhijie.nzj/datasets/princeton-nlp/datasets-for-simcse/wiki1m_for_simcse.txt"
FIRST_MAX_LENGTH=128
FIRST_CHECKPOINT_PATH="lr_5e-05_tua_0.1_2025_07_20_02"
FIRST_CHECKPOINT_ID=10000


PORT=29600

# TEMPERATURE=0.00001
# for SECOND_MAX_LENGTH in 256 512 1024; do
# accelerate launch \
#     --main_process_port $PORT \
#     --num_cpu_threads_per_process 1 \
#     aa_gene_train.py \
#     --model_name $MODEL_NAME \
#     --base_model_name_or_path $BASE_MODEL_PATH \
#     --checkpoint_path ./learn_from_target/$MODEL_NAME/llara_first_last/ml_$FIRST_MAX_LENGTH/$FIRST_CHECKPOINT_PATH/model/checkpoint-${FIRST_CHECKPOINT_ID} \
#     --lora_r 16 \
#     --lora_alpha 32 \
#     --target_modules q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj \
#     --lora_dropout 0.05 \
#     --bias none \
#     --prompt 'This_sentence_:_"*sent_0*"_means_in_one_word:"' \
#     --learning_rate 5e-5 \
#     --per_device_train_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --save_steps 5 \
#     --max_train_steps 1000 \
#     --mode llara_second_last \
#     --pooling_mode last \
#     --output_dir learn_from_target \
#     --max_length $SECOND_MAX_LENGTH \
#     --follow_llara \
#     --dataset_path $DATA_PATH \
#     --tau $TEMPERATURE
# done

SECOND_MAX_LENGTH=128
for TEMPERATURE in 0.0001 1e-6; do

accelerate launch \
    --main_process_port $PORT \
    --num_cpu_threads_per_process 1 \
    aa_gene_train.py \
    --model_name $MODEL_NAME \
    --base_model_name_or_path $BASE_MODEL_PATH \
    --checkpoint_path ./learn_from_target/$MODEL_NAME/llara_first_last/ml_$FIRST_MAX_LENGTH/$FIRST_CHECKPOINT_PATH/model/checkpoint-${FIRST_CHECKPOINT_ID} \
    --lora_r 16 \
    --lora_alpha 32 \
    --target_modules q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj \
    --lora_dropout 0.05 \
    --bias none \
    --prompt 'This_sentence_:_"*sent_0*"_means_in_one_word:"' \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --save_steps 5 \
    --max_train_steps 1000 \
    --mode llara_second_last \
    --pooling_mode last \
    --output_dir learn_from_target \
    --max_length $SECOND_MAX_LENGTH \
    --follow_llara \
    --dataset_path $DATA_PATH \
    --tau $TEMPERATURE
done