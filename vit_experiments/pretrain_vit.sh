#!/bin/bash
### ViT pretraining on ID dataset
# Specify image resolution in RES
# Specify model as 'tvit_tiny_patch8' (ViT-T, depth=12) or 'tvit_small_patch8' (ViT-T+, depth=18)
EPOCHS_NO_AUG=100 # num of epochs
EPOCHS_AUG=60 # num of epochs, 40 (ViT-Tiny 12 layers) and 60 (ViT-Small 18 layers)
NUM_CLASS=100
LR=0.0006  # learning rate, LR=8e-4 (ViT-Tiny 12 layers) and LR=6e-4 (ViT-Small 18 layers)
BS=64 #96 # batch size, BS=96 (ViT-Tiny 12 layers) and BS=64 (ViT-Small 18 layers)
GPU=0,1,2,3 # gpus

## Without Augmentation
for RES in 32 64 128 224
do
    EXPT_NAME=ViT_Small_ImageNet100_${RES}_Patch_8_No_Aug
    SAVE_DIR=./results_small/ViT_Small_ImageNet100_${RES}_Patch_8_No_Aug
    CKPT=checkpoint_${RES}_no_aug.pth

    CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main_no_aug.py \
    --model tvit_small_patch8 \
    --data-path /data/datasets/ImageNet-100 \
    --data-set IMNET \
    --output_dir ${SAVE_DIR} \
    --ckpt_name ${CKPT} \
    --num_class ${NUM_CLASS} \
    --input-size ${RES} \
    --batch-size ${BS} \
    --lr ${LR} \
    --epochs ${EPOCHS_NO_AUG} \
    --warmup-epochs 5 > logs/${EXPT_NAME}.log
done


## With Augmentations
for RES in 32 64 128 224
do
    EXPT_NAME=ViT_Small_ImageNet100_${RES}_Patch_8_Aug
    SAVE_DIR=./results_small/ViT_Small_ImageNet100_${RES}_Patch_8_Aug
    CKPT=checkpoint_${RES}_aug.pth

    CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main_aug.py \
    --model tvit_small_patch8 \
    --data-path /data/datasets/ImageNet-100 \
    --data-set IMNET \
    --output_dir ${SAVE_DIR} \
    --ckpt_name ${CKPT} \
    --num_class ${NUM_CLASS} \
    --input-size ${RES} \
    --batch-size ${BS} \
    --lr ${LR} \
    --epochs ${EPOCHS_AUG} \
    --mixup 0 \
    --cutmix 0 \
    --warmup-epochs 5 > logs/${EXPT_NAME}.log
done
