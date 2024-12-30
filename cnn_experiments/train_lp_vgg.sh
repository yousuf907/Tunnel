#!/bin/bash
### VGG Linear Probing 
## Specify arch as 'vgg13' for vggm-11 and 'vgg19' for vggm-17
# Specify input resolution as 32, 64, 128, 224 in INPUT
# Specify id/ood dataset names in SET
BS=128 # batch size
EPOCH=30 # number of epochs
GPU=0,1,2,3
DATA_DIR=./data/

## Without Augmentations
for INPUT in 32 64 128 224
do
    for SET in imagenet100 ninco imagenet_r200 CIFAR100 CUB200 Aircrafts Pet Flower102 Stl
    do
        EXPT_NAME=LP_vgg11_no_aug_50c_200s_${INPUT}_${SET}
        SAVE_DIR=./lp_results_class_vs_samples
        FEAT_DIR=./features/${SET}_no_aug_${INPUT}
        CKPT_DIR=./vgg11_class_samples
        CKPT=vgg11_50_200_noaug.pth

        CUDA_VISIBLE_DEVICES=${GPU} python -u main_vgg.py \
        --expt_name ${EXPT_NAME} \
        --task lin_probe \
        --no_wandb 0 \
        --data_dir ${DATA_DIR} \
        --save_dir ${SAVE_DIR} \
        --set ${SET} \
        --input_size ${INPUT} \
        --seed 42 \
        --arch 'vgg13' \
        --ckpt_root ${CKPT_DIR} \
        --ckpt_paths ${CKPT} \
        --ckpt_info ${EXPT_NAME} \
        --lr 0.001 \
        --weight_decay 0 \
        --epochs ${EPOCH} \
        --batch_size ${BS} \
        --extract_features 1 \
        --features_per_file 20000 \
        --features_root ${FEAT_DIR}
    done
done


## With Augmentations
for INPUT in 32 64 128 224
do
    for SET in imagenet100 ninco imagenet_r200 CIFAR100 CUB200 Aircrafts Pet Flower102 Stl
    do
        EXPT_NAME=LP_vgg11_aug_50c_200s_${INPUT}_${SET}
        SAVE_DIR=./lp_results_class_samples
        FEAT_DIR=./features/${SET}_aug_${INPUT}
        CKPT_DIR=./vgg11_class_samples
        CKPT=vgg11_50_200_aug.pth

        CUDA_VISIBLE_DEVICES=${GPU} python -u main_vgg.py \
        --expt_name ${EXPT_NAME} \
        --task lin_probe \
        --no_wandb 0 \
        --data_dir ${DATA_DIR} \
        --save_dir ${SAVE_DIR} \
        --set ${SET} \
        --input_size ${INPUT} \
        --seed 42 \
        --arch 'vgg13' \
        --ckpt_root ${CKPT_DIR} \
        --ckpt_paths ${CKPT} \
        --ckpt_info ${EXPT_NAME} \
        --lr 0.001 \
        --weight_decay 0 \
        --epochs ${EPOCH} \
        --batch_size ${BS} \
        --extract_features 1 \
        --features_per_file 20000 \
        --features_root ${FEAT_DIR}
    done
done