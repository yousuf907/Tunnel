import argparse
import time

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data

from tqdm.contrib import tqdm
from pathlib import Path
import wandb

from models.model_utils import get_model
from data.datasets import load_dataset
from utils.utils import checkpoint_summary, AverageMeter, accuracy
from optimizers import get_optimizer



class Trainer:
    # Trainer for transfer learning (retraining)
    def __init__(self, cfg:argparse.Namespace, ckpt_dir: Path=None, ckpt_info: Path=None, model=None) -> None:
        self.cfg = cfg
        self.dataset_name = cfg.set
        # get model
        assert ckpt_dir is not None or model is not None, "Either ckpt_dir or model should be active."
        self.ckpt_info = ckpt_info
        self.model = self.load_model(ckpt_dir, model)
        
        if 'linear' in cfg.task:
            for k,v in self.model.named_parameters():
                if not 'fc' in k:
                    v.requires_grad_(False)
            
            
        self.model.cuda().eval()
        
        # get data
        train_loader, val_loader, test_loader = load_dataset(self.cfg)
        self.test_please = False if test_loader is None else True
        self.loaders = {'train':train_loader, 'val':val_loader, 'test':test_loader}
        
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = get_optimizer(cfg, self.model)
        
        self.losses = AverageMeter("Loss", ":.4f")
        self.top1 = AverageMeter("Acc@1", ":6.4f")
        self.top5 = AverageMeter("Acc@5", ":6.4f")
        self.best_val_acc1 = 0
        self.best_test_acc1 = 0
        
    def load_model(self, ckpt_dir, model=None):
        if model is not None:
            return model
        
        # print(f"loading model from {ckpt_dir}")
        _model = get_model(self.cfg)
        # sys.exit()
        _model_dict = _model.state_dict()

        #print("\nThis is the defined arch:")
        #for param_tensor in _model.state_dict():
        #    print(param_tensor, "\t", _model.state_dict()[param_tensor].size())

        
        ckpt = torch.load(ckpt_dir)
        for k in ckpt.keys():
            if 'state_dict' in k:
                _state_dict_key = k

        ### Remove module
        state_dict = ckpt[_state_dict_key]
        for k in list(state_dict.keys()):
            # retain model only
            if k.startswith('module'):
                # remove prefix
                state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        ckpt[_state_dict_key] = state_dict
        

        #print("\nThis is the pre-trained ckpt:")
        #for param_tensor in ckpt[_state_dict_key]:
        #    print(param_tensor, "\t", ckpt[_state_dict_key][param_tensor].size())

        # if self.cfg.debug:
        #     print(f"checkpoint keys:\n{ckpt.keys()}")
        layers_to_drop = [k for k in ckpt[_state_dict_key].keys() if 'layer.' in k or 'mask' in k]
        #print(layers_to_drop)
        if 'transfer' in self.cfg.task:
            layers_to_drop.extend(['fc.weight', 'fc.bias'])
        
        for k in layers_to_drop:
            ckpt[_state_dict_key].pop(k)
        
        #print(ckpt[_state_dict_key].keys())
            
        _model_dict.update(ckpt[_state_dict_key])
        
        _model.load_state_dict(_model_dict)

        model_state_dict = _model.state_dict()
        for k in ckpt[_state_dict_key]:
            #if not k.startswith('fc.'):
            assert torch.equal(model_state_dict[k].cpu(), ckpt[_state_dict_key][k].cpu()), k
        print("\nSuccessfully performed sanity-check on loading pre-trained ckpt")
        
        if 'best_val_acc' in ckpt:
            self.best_val_acc1 = ckpt['best_val_acc']
            if 'best_test_acc' in ckpt.keys():
                self.best_test_acc1 = ckpt['best_test_acc']

            checkpoint_summary(ckpt_dir, ckpt)
        # print(f"checkpoint's best accuracy is {ckpt['best_val_acc']}")
        return _model
        
    def compute_forward(self, inputs, targets):
        return self.model(inputs)
    
    def ce_forward(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        if outputs.size(-1) >= 5:
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        else:
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            acc5 = torch.zeros_like(acc1)
        
        return loss, acc1, acc5
            
    # we need self.criterion, self.model, and self.kdloss for this
    def compute_objectives(self, inputs, targets): # returns loss
        """Need self.criterion and self.model to do certain behavior for this function
        """
        loss, acc1, acc5 = self.ce_forward(inputs, targets)
        
        self.losses.update(loss.item(), inputs.size(0))
        self.top1.update(acc1.item(), inputs.size(0))
        self.top5.update(acc5.item(), inputs.size(0))
        
        return loss
           
    def fit_batch(self, inputs, outputs):
        """Fit one batch, override to do multiple updates.
        The default implementation depends on a few methods being defined
        with a particular behavior:
        * ``compute_objectives()``
        Also depends on having optimizers passed at initialization.
        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        Returns
        -------
        detached loss
        num of correct predictions
        """
        # Managing automatic mixed precision
        if False: #self.cfg.auto_mix_prec:
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = self.compute_objectives(inputs, outputs)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.compute_objectives(inputs, outputs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.detach().cpu()
    
    
    def train(self, epoch, progressbar=True):
        self.model.train()
        train_loader = self.loaders['train']
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()
        

        _train_start_time = time.time()
        
        with tqdm(
            train_loader,
            disable=not progressbar
            ) as t:
            for batch_idx, (inputs, targets) in enumerate(t):
                inputs, targets = inputs.cuda(),targets.long().squeeze().cuda()
                loss = self.fit_batch(inputs, targets)
                
                t.set_postfix(train_loss=self.losses.avg)
                if self.cfg.debug and batch_idx == self.cfg.debug_batches - 1:
                    break
                
        train_summary = {
                f'Train/Loss {self.ckpt_info}':self.losses.avg,
                f'Train/Acc1 {self.ckpt_info}':self.top1.avg,
                f'Train/Acc5 {self.ckpt_info}':self.top5.avg,
                f'Train/epoch {self.ckpt_info}': epoch,
                f'Train/time {self.ckpt_info}': time.time() - _train_start_time
                }                
        if not self.cfg.no_wandb and not self.cfg.debug:
            wandb.log(train_summary)

    def evaluate(self, mode, epoch, progressbar=True):
        self.model.eval()
        
        val_loader = self.loaders[mode]
        
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()
        
        _val_start_time = time.time()
        with torch.no_grad():
            with tqdm(
            val_loader,
            disable=progressbar,
            ) as t:
                for batch_idx, (inputs, targets) in enumerate(t):
                    inputs, targets = inputs.cuda(),targets.long().squeeze().cuda()
                    _ = self.compute_objectives(inputs, targets)
                    t.set_postfix(test_loss=self.top1.avg)
                    if self.cfg.debug and batch_idx == self.cfg.debug_batches - 1:
                        break
    
        # ON Val end
        cur_val_loss = self.losses.avg
        cur_val_acc1 = self.top1.avg
        cur_val_acc5 = self.top5.avg

        if mode == 'val':
            is_best_val = cur_val_acc1 > self.best_val_acc1
            self.best_val_acc1 = max(self.best_val_acc1, cur_val_acc1)
            val_summary = {
                        f'Val/Loss {self.ckpt_info}':cur_val_loss,
                        f'Val/Acc1 {self.ckpt_info}':cur_val_acc1,
                        f'Val/Acc5 {self.ckpt_info}':cur_val_acc5,
                        f'Best/Best Val Acc1 {self.ckpt_info}': self.best_val_acc1,
                        f'Val/epoch': epoch,
                        }
            
        if mode == 'test':
            is_best_test = cur_val_acc1 > self.best_test_acc1
            self.best_test_acc1 = max(self.best_test_acc1, cur_val_acc1)
            val_summary = {
                f'TEST/Loss {self.ckpt_info}':cur_val_loss,
                f'TEST/Acc1 {self.ckpt_info}':cur_val_acc1,
                f'TEST/Acc5 {self.ckpt_info}':cur_val_acc5,
                f'Best/Best TEST Acc1 {self.ckpt_info}': self.best_val_acc1,
                f'TEST/epoch': epoch,
                }
            
        if not self.cfg.no_wandb and not self.cfg.debug:
            wandb.log(val_summary)
            
            
    def fit(self):
        for e in range(self.cfg.epochs):
            self.train(e)
            self.evaluate('val', e)
            if self.test_please:
                self.evaluate('test', e)