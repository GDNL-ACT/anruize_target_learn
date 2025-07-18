````markdown
# 项目说明

## 环境配置
- CUDA 版本：12.8
- Python 版本：3.12.9
- 交互配置：使用 `accelerate config` 进行配置

---

## 一阶段训练执行

请根据实际情况调整 batch size 和 max length 参数，然后执行以下脚本开始一阶段训练：

```bash
./script/mistral_llara_first_train.sh
````

训练结果将保存在以下路径（不用复制，只需要在最上面填变量FIRST_MAX_LENGTH、FIRST_CHECKPOINT_PATH、FIRST_CHECKPOINT_ID的值）：

```
./learn_from_target/$MODEL_NAME/llara_first/ml_$FIRST_MAX_LENGTH/$FIRST_CHECKPOINT_PATH/model/checkpoint-${FIRST_CHECKPOINT_ID}
```

---

## 二阶段训练准备

在执行二阶段训练前，请先将一阶段训练的相关信息填写到：

```
./script/mistral_llara_second_train.sh
```

主要需要填写以下三个参数：

```bash
FIRST_MAX_LENGTH=128
FIRST_CHECKPOINT_PATH="lr_5e-05_tua_0.1_2025_07_18_15"
FIRST_CHECKPOINT_ID=5
```

填写完成后，设置本阶段的 batch size 和 max length，最后执行：

```bash
./script/mistral_llara_second_train.sh
```

