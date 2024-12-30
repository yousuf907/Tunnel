#!/usr/bin/env bash
## Pretrain DNNs without augmentation
DATA_DIR=/data/datasets/ImageNet-100
NUM_CLASS=100
NUM_EPOCHS=100
GPU=0,1,2,3

for RES in 32 64 128 224
do
    EXPT_NAME=ResNet18_ImageNet100_${RES}_No_Aug
    SAVE_DIR=./results_resnet18
    CKPT=resnet18_checkpoint_${RES}_no_aug.pth

    CUDA_VISIBLE_DEVICES=${GPU} python -u pretrain_no_aug.py \
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


