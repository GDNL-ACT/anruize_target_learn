CUDA Version: 12.8 
Python 3.12.9
交互配置accelerate config


一阶段训练执行（batch size 和 max length 需要根据情况设置一下）：
./script/mistral_llara_first_train.sh
训练结果存在路径：
./learn_from_target/$MODEL_NAME/llara_first/ml_$FIRST_MAX_LENGTH/$FIRST_CHECKPOINT_PATH/model/checkpoint-${FIRST_CHECKPOINT_ID}


二阶段训练之前
把第一阶段的训练信息填在./script/mistral_llara_second_train.sh文件里
主要是以下三个
FIRST_MAX_LENGTH=128
FIRST_CHECKPOINT_PATH="lr_5e-05_tua_0.1_2025_07_18_15"
FIRST_CHECKPOINT_ID=5
填好后，设置本阶段的（batch size 和 max length ）最后执行
./script/mistral_llara_second_train.sh
