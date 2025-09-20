#!/bin/bash

# Set the NCCL timeout to 600000 milliseconds (10 minutes)
export NCCL_TIMEOUT=60000
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1



MODEL_NAME="mistral-7b"
BASE_MODEL_PATH="/data/LLM/llama3/llama3-8b-instruct"

# CHECKPOINT_ID_FIRST_STEP=20
# CHECKPOINT_PATH_FISRT_STEP="lr_5e-05_tua_0.1_2025_07_26_12"



MODE="noprompt_meanpool_lmhead"
POOLING_MODEL="mean"
MAX_LENGTH=128
BATCH_SIZE=8

FILE_NAME="learn_target_pretrain"
PEFT_MODEL_PATH_FIRST_STEP="./$FILE_NAME/llama_mean_1step_bid"
PEFT_MODEL_PATH="./$FILE_NAME/llama_mean_2step_bid"
OUTPUT_PATH="./$FILE_NAME/llama_mean_2step_bid"

echo "Start checkpoint-$CHECKPOINT_ID"
echo "OUTPUT_PATH-$OUTPUT_PATH"
python aa_gene_test_all.py \
    --model_name $MODEL_NAME \
    --base_model_name_or_path $BASE_MODEL_PATH \
    --pre_checkpoint_path $PEFT_MODEL_PATH_FIRST_STEP \
    --peft_model_name_or_path $PEFT_MODEL_PATH \
    --mode $MODE \
    --pooling_mode $POOLING_MODEL \
    --output_dir $OUTPUT_PATH \
    --batch_size $BATCH_SIZE \
    --follow_llara \
    --enable_bidirectional