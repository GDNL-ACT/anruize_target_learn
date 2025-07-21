#!/bin/bash

# Set the NCCL timeout to 600000 milliseconds (10 minutes)
export NCCL_TIMEOUT=60000
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

 
MODEL_NAME="llama3-8b"
BASE_MODEL_PATH="/data/LLM/llama3/llama3-8b-instruct"
CHECKPOINT_ID_FIRST_STEP=5
CHECKPOINT_PATH_FISRT_STEP="lr_5e-05_tua_0.1_2025_07_21_11"
CHECKPOINT_PATH="lr_0.0005_tua_1e-05_2025_07_21_11"
MODE="llara_second_last"
POOLING_MODEL="last"
MAX_LENGTH=128
BATCH_SIZE=8


for CHECKPOINT_ID in $(seq 10 10 100); do
    PEFT_MODEL_PATH_FIRST_STEP="./learn_from_target/$MODEL_NAME/llara_first_$POOLING_MODEL/ml_$MAX_LENGTH/$CHECKPOINT_PATH_FISRT_STEP/model/checkpoint-$CHECKPOINT_ID_FIRST_STEP"
    PEFT_MODEL_PATH="./learn_from_target/$MODEL_NAME/$MODE/ml_$MAX_LENGTH/$CHECKPOINT_PATH/model/checkpoint-$CHECKPOINT_ID"
    OUTPUT_PATH="./learn_from_target/$MODEL_NAME/$MODE/ml_$MAX_LENGTH/$CHECKPOINT_PATH/model/checkpoint-$CHECKPOINT_ID"
    # OUTPUT_PATH="./learn_from_target/$MODEL_NAME/$MODE/ml_$MAX_LENGTH/$CHECKPOINT_PATH/model/checkpoint-$CHECKPOINT_ID/use_instructions"
    echo "Start checkpoint-$CHECKPOINT_ID"
    echo "OUTPUT_PATH-$OUTPUT_PATH"
    python aa_gene_test.py \
        --model_name $MODEL_NAME \
        --base_model_name_or_path $BASE_MODEL_PATH \
        --pre_checkpoint_path $PEFT_MODEL_PATH_FIRST_STEP \
        --peft_model_name_or_path $PEFT_MODEL_PATH \
        --mode $MODE \
        --pooling_mode $POOLING_MODEL \
        --output_dir $OUTPUT_PATH \
        --batch_size $BATCH_SIZE \
        --follow_llara
        # --use_instructions
done