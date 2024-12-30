import torch
import wandb
import torch.optim
import torch.utils.data
from tqdm.contrib import tqdm
from feature_loader import load_features
from utils.utils import knn_accuracy, AverageMeter
import numpy as np
import random

class KNN_Probe:
    def __init__(self, cfg, feature_dir, ckpt_info):
        self.feature_dir = feature_dir
        self.ckpt_info = ckpt_info
        self.cfg = cfg
        self.train_loader, self.val_loader, self.test_loader = load_features(cfg, feature_dir)
        
        # Which split is the target for knn? (val, test or both)
        self.test_modes = []
        if self.val_loader is not None:
            self.test_modes.append('val')
        if self.test_loader is not None:
            self.test_modes.append('test')
        assert len(self.test_modes) > 0, "Must test on something"
        # How many layers do we wanna probe on
        self.len_layers = self.train_loader.dataset.len_layers()    
        
        # knn usefule variables
        self.K = cfg.knn_K
        
    
    def _probe(self, train_loader, test_loader, layer_name, progress_bar=False):
        total_test_processed = 0
        max_k = max(self.K)
        with tqdm(test_loader, disable=not progress_bar) as t:
            for test_idx, (test_features, test_targets) in enumerate(t):
                test_bs = test_features.size(0)
                # Top K best similarity scores for each test sample
                test_features, test_targets = test_features.cuda(), test_targets.cuda()
                top_scores = torch.zeros((test_bs, max_k), device=torch.device('cuda')) + 1000000
                top_labels = torch.zeros((test_bs, max_k), device=torch.device('cuda'), dtype=torch.int32) - 1
                for train_idx, (train_features, train_targets) in enumerate(train_loader):
                    train_features, train_targets = train_features.cuda(), train_targets.cuda()
                    scores = torch.norm(
                        test_features[:, None, :] - train_features[None, :, :], dim=-1
                    )                                                               # [test_bs, train_bs] 
                    scores, inds = scores.topk(max_k, dim=-1, largest=False, )      # [test_bs, maxk]
                    train_targets = train_targets.view(1, -1).repeat(test_bs, 1)    # [test_bs, train_bs]
                    train_targets = train_targets.gather(1, inds)                   # [test_bs, maxk]
                    
                    scores = torch.cat([scores, top_scores], dim=-1)                # [test_bs, maxk(top overall before the current batch)+maxk(top of the current training batch)]
                    labels = torch.cat([train_targets, top_labels], dim=-1)         # [test_bs, maxk(top overall before the current batch)+maxk(top of the current training batch)]
                    top_scores, indices = scores.topk(k=max_k, dim=-1, largest=False) # [test_bs, maxk]
                    top_labels = labels.gather(1, indices)                            # [test_bs, maxk]
                    # top_scores and labels are calculated now
                    
                accs = knn_accuracy(top_labels, test_targets, topk=self.K)          # batch accuracy
                for k in self.K:
                    # update meters
                    self.acc_meters[k].update(accs[k], test_targets.size(0))
                
                t.set_postfix(train_loss=self.acc_meters[k].avg)
        print("Layer:", layer_name, "Feature Dim:", test_features.shape)
                
    def on_probe_start(self, layer_name):
        # initialize meters
        self.acc_meters = {k:AverageMeter(f"Acc@{k}", "6:4") for k in self.K}

    def on_probe_end(self, layer_idx, mode):
        self.accs = {k:meter.avg for k, meter in self.acc_meters.items()}
        for k, meter in self.acc_meters.items():
            meter.reset()
            
            # Logging
            if not self.cfg.no_wandb:
                model_config =  'OOD_vs_ID' #self.ckpt_info #self.feature_dir.parts[-3]
                info = 'ResNet18_Linear_Probe'
                wandb.log({
                    #'l_idx': layer_idx,
                    #f'{self.ckpt_info}': self.accs[k]
                    f'{info}': self.accs[k]
                })
        #print(self.accs[k])
        return self.accs[k]
    
    '''
    def on_probe_end(self, layer_idx, mode):
        self.accs = {k:meter.avg for k, meter in self.acc_meters.items()}
        for k, meter in self.acc_meters.items():
            meter.reset()
            
            # Logging
            if not self.cfg.no_wandb:
                model_config =  self.ckpt_info #self.feature_dir.parts[-3]
                wandb.log({
                    'l_idx': layer_idx,
                    f'{k}/{self.ckpt_info}': self.accs[k]
                })
    '''
    
    @torch.no_grad()
    def probe(self):
        lp_acc=[]
        for l_idx in range(0, self.len_layers):
            layer_name = self.train_loader.dataset.set_layer(l_idx)
            print(f'processing layer {layer_name}')
            self.on_probe_start(layer_name)
            for test_mode in self.test_modes:
                if test_mode == 'val':
                    self.val_loader.dataset.set_layer(l_idx)
                    self._probe(self.train_loader, self.val_loader, layer_name)
                    top1_acc = self.on_probe_end(l_idx, 'val')
                    lp_acc = np.append(lp_acc, top1_acc.cpu().numpy())
                elif test_mode == 'test':
                    self.test_loader.dataset.set_layer(l_idx)
                    self._probe(self.train_loader, self.test_loader, layer_name)
                    self.on_probe_end(l_idx, 'test')

        return lp_acc
            