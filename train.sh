#!/bin/bash

# EMSB模型训练脚本
# 基于launch.json中的配置参数

# 训练参数
LOG_DIR="logs/"
NAME="emsb_stl_1w5"
EMA=0.99
LR="5e-5"
LR_BRIGHTNESS="5e-5"
SCHEDULER_TYPE="constant"
BATCH_SIZE=2
ACCUMULATE_GRAD_BATCHES=16
IMAGE_SIZE=256
MAX_STEPS=15000
NUM_WORKERS=4
DEVICES="0"
VAL_EVERY_N_STEPS=500
SAVE_EVERY_N_STEPS=1000
BRIGHTNESS_TYPE="stl" # "naive" "l2""stl""vae"
BRIGHTNESS_CHANNEL=1 # 1 3

VAL_EVERY_N_BATCHES=$((VAL_EVERY_N_STEPS * ACCUMULATE_GRAD_BATCHES))

# 构建训练命令
.venv/bin/python train.py \
    --log_dir $LOG_DIR \
    --name $NAME \
    --use_fp16 \
    --ema $EMA \
    --lr $LR \
    --lr_brightness $LR_BRIGHTNESS \
    --scheduler_type $SCHEDULER_TYPE \
    --batch_size $BATCH_SIZE \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --image_size $IMAGE_SIZE \
    --max_steps $MAX_STEPS \
    --num_workers $NUM_WORKERS \
    --devices $DEVICES \
    --val_every_n_batches $VAL_EVERY_N_BATCHES \
    --save_every_n_steps $SAVE_EVERY_N_STEPS \
    --brightness_net \
    --brightness_type $BRIGHTNESS_TYPE \
    --brightness_channel $BRIGHTNESS_CHANNEL \
    # --blur_loss 0.5 

echo "训练完成！"
