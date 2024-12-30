from config import Config
from knn_probe import KNN_Probe
from linear_probe2 import Linear_Probe, _set_seed
from transfer import Trainer as TransferTrainer
import wandb
import numpy as np
import os
import shutil

if __name__ == "__main__":
    cfg = Config().parse(None)

    if cfg.set == "ninco":
        from feature_extractor_ninco_resnet import Feature_Extractor ## For ResNet architectures and NINCO OOD dataset
    else:
        from feature_extractor import Feature_Extractor ## For ResNet architectures and all other experiments

    # Feature extraction:
    if cfg.task == 'transfer_linear':
        proj_name = 'transfer'
    if cfg.task == 'knn_probe':
        proj_name = 'KNN_probe'
    if cfg.task == 'lin_probe':
        proj_name = 'Linear_probe'
    proj_name = proj_name if cfg.wandb_project_name=='' else cfg.wandb_project_name

    #set seed
    _set_seed(cfg.seed)

    if not cfg.no_wandb:
        wandb.init(project=proj_name, entity=cfg.wandb_entity, name=cfg.expt_name, config=cfg, mode = cfg.wandb_offline)

    if cfg.extract_features:
        for ckpt_path, feature_path in zip(cfg.ckpt_full_paths, cfg.features_full_paths):
            Feature_Extractor(cfg, ckpt_path, feature_path).extract()

    for ckpt_path, feature_path, ckpt_info in zip(cfg.ckpt_full_paths, cfg.features_full_paths, cfg.ckpt_info):
        if cfg.task == 'knn_probe':
            lp_acc = KNN_Probe(cfg, feature_path, ckpt_info).probe()
        elif cfg.task == 'lin_probe':
            lp_acc, preds_all, gt = Linear_Probe(cfg, feature_path, ckpt_info).probe()
        elif 'transfer' in cfg.task:
            TransferTrainer(cfg, ckpt_path, ckpt_info).fit()

    exp_dir = cfg.save_dir
    filename = cfg.expt_name + '.npy'
    #filename2 = cfg.expt_name + '_preds.npy'
    #filename3 = cfg.expt_name + '_gt.npy'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    np.save(os.path.join(exp_dir, filename), lp_acc)
    #np.save(os.path.join(exp_dir, filename2), preds_all)
    #np.save(os.path.join(exp_dir, filename3), gt)
    print("File is saved")
    print("Linear Probe Accuracy:", lp_acc)

    ## Remove saved features
    if os.path.isdir(cfg.features_root):
        shutil.rmtree(cfg.features_root)  # remove dir and all contains
        print("Saved features are removed")
