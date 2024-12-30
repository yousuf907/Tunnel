#!/bin/bash
### Linear Probing with ViT
# Specify image resolution in INPUT_SIZE
# Specify model as 'tvit_tiny_patch8' (ViT-T, depth=12) or 'tvit_small_patch8' (ViT-T+, depth=18)
SAVE_DIR=./lp_results_vit_small
EPOCHS=30
GPU=0,1,2,3 # gpus
DATA_DIR=./data/

## With Augmentation
for INPUT_SIZE in 32 64 128 224
do
    for DATASET in 'IMNET' 'IMNETR' 'NINCO' 'Aircraft' 'CIFAR' 'Flower102' 'Pet' 'CUB200' 'Stl'
    do
        EXPT_NAME=LP_ViT_Small_aug_${INPUT_SIZE}_${DATASET}
        CKPT=./results_small/ViT_Small_IN100_${INPUT_SIZE}_Patch_8_Aug/best_checkpoint_${INPUT_SIZE}_aug.pth

        CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --use_env main_lp.py \
        --expt_name ${EXPT_NAME} \
        --model tvit_small_patch8 \
        --data-set ${DATASET} \
        --data-path ${DATA_DIR} \
        --output_dir ${SAVE_DIR} \
        --input-size ${INPUT_SIZE} \
        --batch-size 512 \
        --lr 0.01 \
        --epochs ${EPOCHS} \
        --seed 0 \
        --warmup-epochs 0 \
        --smoothing 0.1 \
        --finetune ${CKPT} \
        --mixup 0.0 \
        --cutmix 0.0 > logs/${EXPT_NAME}.log
    done
done

## Without Augmentation
for INPUT_SIZE in 32 64 128 224
do
    for DATASET in 'IMNET' 'IMNETR' 'NINCO' 'Aircraft' 'CIFAR' 'Flower102' 'Pet' 'CUB200' 'Stl'
    do
        EXPT_NAME=LP_ViT_Small_no_aug_${INPUT_SIZE}_${DATASET}
        CKPT=./results_small/ViT_Small_IN100_${INPUT_SIZE}_Patch_8_No_Aug/best_checkpoint_${INPUT_SIZE}_no_aug.pth

        CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --use_env main_lp.py \
        --expt_name ${EXPT_NAME} \
        --model tvit_small_patch8 \
        --data-set ${DATASET} \
        --data-path ${DATA_DIR} \
        --output_dir ${SAVE_DIR} \
        --input-size ${INPUT_SIZE} \
        --batch-size 512 \
        --lr 0.01 \
        --epochs ${EPOCHS} \
        --seed 0 \
        --warmup-epochs 0 \
        --smoothing 0.1 \
        --finetune ${CKPT} \
        --mixup 0.0 \
        --cutmix 0.0 > logs/${EXPT_NAME}.log
    done
done