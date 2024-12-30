#!/bin/bash
## Pretrain DNNs with augmentation
DATA_DIR=/data/datasets/ImageNet-100
NUM_EPOCHS=80 # 80 epochs for ResNet18 Aug, 100 epochs for ResNet34 Aug
NUM_CLASS=100
GPU=0,1,2,3 # gpu

for RES in 32 64 128 224
do
    EXPT_NAME=ResNet18_ImageNet100_${RES}_Aug
    SAVE_DIR=./results_resnet18
    CKPT=resnet18_checkpoint_${RES}_aug.pth

    CUDA_VISIBLE_DEVICES=${GPU} python -u pretrain_aug.py \
    --data ${DATA_DIR} \
    --save_dir ${SAVE_DIR} \
    --num_classes ${NUM_CLASS} \
    --image_size ${RES} \
    --lr 0.01 \
    --wd 5e-2 \
    --epochs ${NUM_EPOCHS} \
    -b 2048 \
    -p 250 \
    --ckpt_file ${EXPT_NAME}_${NUM_CLASS}.pth \
    --expt_name ${EXPT_NAME} > logs/${EXPT_NAME}.log
done

