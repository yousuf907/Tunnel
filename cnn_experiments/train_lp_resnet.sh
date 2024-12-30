#!/bin/bash
### ResNet Linear Probing 
## Specify 'ResNet18' and 'ResNet34' in arch
# Specify input resolution as 32, 64, 128, 224 in INPUT
# Specify id/ood dataset names in SET
BS=256 # batch size
EPOCH=30
GPU=0,1,2,3 # gpu
SAVE_DIR=./lp_results_resnet18
DATA_DIR=./data/


### Without Augmentations
for INPUT in 32 64 128 224
do
    for SET in imagenet100 ninco imagenet_r200 CIFAR100 CUB200 Aircrafts Pet Flower102 Stl
    do
        EXPT_NAME=LP_resnet18_no_aug_${INPUT}_${SET}
        FEAT_DIR=./features/${SET}_no_aug_${INPUT}
        CKPT_DIR=./results_resnet
        CKPT=best_RN18_IN100_${INPUT}_No_Aug_100.pth

        CUDA_VISIBLE_DEVICES=${GPU} python -u main_resnet.py \
        --expt_name ${EXPT_NAME} \
        --task lin_probe \
        --no_wandb 0 \
        --data_dir ${DATA_DIR} \
        --save_dir ${SAVE_DIR} \
        --set ${SET} \
        --input_size ${INPUT} \
        --seed 0 \
        --arch 'ResNet18' \
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


### With Augmentations
for INPUT in 32 64 128 224
do
    for SET in imagenet100 ninco imagenet_r200 CIFAR100 CUB200 Aircrafts Pet Flower102 Stl
    do
        EXPT_NAME=LP_resnet18_aug_${INPUT}_${SET}
        FEAT_DIR=./features/${SET}_aug_${INPUT}
        CKPT_DIR=./results_resnet
        CKPT=best_RN18_IN100_${INPUT}_Aug_100.pth

        CUDA_VISIBLE_DEVICES=${GPU} python -u main_resnet.py \
        --expt_name ${EXPT_NAME} \
        --task lin_probe \
        --no_wandb 0 \
        --data_dir ${DATA_DIR} \
        --save_dir ${SAVE_DIR} \
        --set ${SET} \
        --input_size ${INPUT} \
        --seed 0 \
        --arch 'ResNet18' \
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

