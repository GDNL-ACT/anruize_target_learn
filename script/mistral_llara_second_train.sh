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
BASE_MODEL_PATH="/data/LLM/mistral/mistral-7b-v0_2-chat"
FIRST_MAX_LENGTH=128
FIRST_CHECKPOINT_PATH="lr_5e-05_tua_0.1_2025_07_18_15"
FIRST_CHECKPOINT_ID=5

SECOND_MAX_LENGTH=128


PORT=29600  # 你可以根据实际需要修改端口号
accelerate launch \
    --main_process_port $PORT \
    --num_cpu_threads_per_process 1 \
    aa_gene_train.py \
    --model_name $MODEL_NAME \
    --base_model_name_or_path $BASE_MODEL_PATH \
    --checkpoint_path ./learn_from_target/$MODEL_NAME/llara_first/ml_$FIRST_MAX_LENGTH/$FIRST_CHECKPOINT_PATH/model/checkpoint-${FIRST_CHECKPOINT_ID} \
    --lora_r 16 \
    --lora_alpha 32 \
    --target_modules q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj \
    --lora_dropout 0.05 \
    --bias none \
    --prompt 'This_sentence_:_"*sent_0*"_means_in_one_word:"' \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --save_steps 5 \
    --max_train_steps 100 \
    --mode llara_second \
    --pooling_mode last \
    --output_dir learn_from_target \
    --max_length 128 \
    --follow_llara \
    --dataset_path './data/wiki1m_for_simcse_mini.txt' \
    --tau 0.1
