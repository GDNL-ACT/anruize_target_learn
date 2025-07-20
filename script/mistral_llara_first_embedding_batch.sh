#!/bin/bash

# 设置必要路径和参数
BASE_MODEL_NAME="mistral-7b"
BASE_MODEL_PATH="/data/LLM/mistral/mistral-7b-v0_2-chat"
DATASET_PATH="cfli/pretrain_wiki"
MODE="llara_first"
POOLING_MODE="last"
FIRST_MAX_LENGTH=128
PROMPT='This_sentence_:_"*sent_0*"_means_in_one_word:'
BATCH_SIZE=32
CHECKPOINT_PATH="lr_5e-05_tua_0.1_2025_07_18_15"

for CHECKPOINT_ID in $(seq 10 10 100); do
PORT=29600 
accelerate launch \
    --main_process_port $PORT \
    --num_cpu_threads_per_process 1 \
    embedding_save.py \
    --model_name "$BASE_MODEL_NAME" \
    --base_model_name_or_path "$BASE_MODEL_PATH" \
    --checkpoint_path "./learn_from_target/$BASE_MODEL_NAME/$MODE/ml_$FIRST_MAX_LENGTH/$CHECKPOINT_PATH/model/checkpoint-$CHECKPOINT_ID" \
    --dataset_path "$DATASET_PATH" \
    --mode "$MODE" \
    --pooling_mode "$POOLING_MODE" \
    --max_length $FIRST_MAX_LENGTH \
    --prompt "$PROMPT" \
    --batch_size $BATCH_SIZE
done