# --------------------------------------------------------
# The following code is based on 2 codebases:
## (1) A-ViT
# https://github.com/NVlabs/A-ViT
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
## (2) DeiT
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc. All rights reserved.
# The code is modified to accomodate ViT training
# --------------------------------------------------------

"""
Train and eval functions used in main_lp.py for ViT linear probe training and eval. NB: ViT is trained without class token!
"""

import math
import sys
from typing import Iterable, Optional
import torch
import torch.nn as nn
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from losses import DistillationLoss
import utils
from timm.utils.utils import *
import numpy as np
import os
from utils import RegularizationLoss
import pickle
import heapq, random
from PIL import Image
import cv2
import json


def train_one_epoch(model: torch.nn.Module, linear_classifier: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None, tf_writer=None):

    #model.train(set_training_mode)
    model.eval()
    #linear_classifier.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500
    batch_cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        #targets = targets - 100 # for OOD ImageNet 100-199 class ids

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        #print("shape of input image:", samples.shape)

        # forward
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(samples)
            #print(len(intermediate_output)) # 12

            output = []
            for each_layer in intermediate_output:
                #cls_rep = each_layer[:, 0]
                #mean_rep = torch.mean(each_layer[:, 1:], dim=1)
                mean_rep = torch.mean(each_layer, dim=1)
                #output.append(torch.cat((cls_rep, mean_rep), dim=-1).float())
                ## excluding class token
                output.append(mean_rep.float())

        with torch.cuda.amp.autocast():
            # compute cross entropy loss
            #loss = 0
            for each_output, each_classifier, each_optimizer in zip(output, linear_classifier, optimizer):
                #loss += nn.CrossEntropyLoss()(each_output, targets)
                #loss += criterion(each_output, targets)
                each_classifier.train()
                each_output = each_classifier(each_output)
                each_optimizer.zero_grad()
                loss = criterion(each_output, targets)
                loss.backward()
                # step
                each_optimizer.step()

        batch_cnt += 1

        loss_value = loss.item()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=each_optimizer.param_groups[0]["lr"])

        if tf_writer is not None and torch.cuda.current_device()==0:
            cnt_token=0.0
            cnt_token_diff=0.0
            ponder_loss_token=0.0
            distr_prior_loss=0.0
            if batch_cnt % print_freq == 0:
                tf_writer.add_scalar('train/lr', each_optimizer.param_groups[0]["lr"], batch_cnt)
                tf_writer.add_scalar('train/loss', loss_value, batch_cnt)
                tf_writer.add_scalar('train/cnt_token_mean', float(np.mean(cnt_token)), batch_cnt)
                tf_writer.add_scalar('train/cnt_token_max', float(np.max(cnt_token)), batch_cnt)
                tf_writer.add_scalar('train/cnt_token_min', float(np.min(cnt_token)), batch_cnt)
                tf_writer.add_scalar('train/avg_cnt_token_diff', float(np.mean(cnt_token_diff)), batch_cnt)
                tf_writer.add_scalar('train/ponder_loss_token', ponder_loss_token, batch_cnt)
                tf_writer.add_scalar('train/expected_depth_ratio', float(np.mean(cnt_token/12.)), batch_cnt)
                #if args.distr_prior_alpha > 0.:
                #    tf_writer.add_scalar('train/distr_prior_loss', distr_prior_loss.item(), batch_cnt)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, linear_classifier, device, epoch, tf_writer=None, args=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    #linear_classifier.eval()

    cnt_token, cnt_token_diff = None, None

    preds = torch.zeros((args.num_layers, len(data_loader.dataset)), dtype=torch.float32)
    all_lbls = torch.zeros((len(data_loader.dataset)), dtype=torch.float32)
    start_ix = 0

    for images, target in metric_logger.log_every(data_loader, 50, header):
        #target = target - 100 # for OOD ImageNet 100-199 class ids
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # forward
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(images)

            output = []
            for each_layer in intermediate_output:
                #cls_rep = each_layer[:, 0]
                #mean_rep = torch.mean(each_layer[:, 1:], dim=1)
                mean_rep = torch.mean(each_layer, dim=1)
                #output.append(torch.cat((cls_rep, mean_rep), dim=-1).float())
                output.append(mean_rep.float())
            #all_output = linear_classifier(output)

        end_ix = start_ix + len(target)
        i=0
        for each_output, each_classifier in zip(output, linear_classifier):
            i += 1  
            each_classifier.eval()
            each_output = each_classifier(each_output)
            loss = criterion(each_output, target)
            acc1, acc5 = accuracy(each_output, target, topk=(1, 5))
            
            preds[i-1, start_ix:end_ix] = each_output.argmax(dim=1)
            all_lbls[start_ix:end_ix] = target.squeeze()

            batch_size = images.shape[0]
            post_str = '_layer%d' % i
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1' + post_str].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5' + post_str].update(acc5.item(), n=batch_size)

        start_ix = end_ix

    eval_results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    updated_results = {}
    for key in eval_results:
        if '_' in key:
            this_key, classifier_idx = key.split('_')

            if classifier_idx not in updated_results:
                updated_results[classifier_idx] = {}

            updated_results[classifier_idx][this_key] = eval_results[key]

    return updated_results, preds, all_lbls



@torch.no_grad()
def evaluate_ckpt(data_loader, model, device, epoch, args=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    cnt_token, cnt_token_diff = None, None

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    # snippet for merging and visualization
    h_min = max(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def merge_image(im1, im2):
    # snippet for merging and visualization
    h_margin = 54
    v_margin = 80
    im2 = im2[h_margin+5:480-h_margin, v_margin:640-v_margin]
    return hconcat_resize_min([im1, im2])



@torch.no_grad()
def visualize(data_loader, model, device, epoch, tf_writer=None, args=None):
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt
    from PIL import Image

    # this snipet visualize the token depth distribution of an avit model
    # more particular, it saves the image with the largset token depth std. per imagenet class
    # in validation set.

    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Visualize:'

    # switch to evaluation mode
    model.eval()
    save_image = True

    # amid imagenet class separation for best visualization, assert batch size is 10
    # such that no validation images overlap in classes
    assert args.batch_size==50
    class_set = set()

    for images, target in metric_logger.log_every(data_loader, 100, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        cnt_token = model.module.counter_token.data.cpu().numpy()

        # this tries to save images
        if save_image:

            cnt_token_std_lst = np.std(cnt_token, axis=-1)
            value = np.max(cnt_token_std_lst)
            key = np.argmax(cnt_token_std_lst)

            # this part fetches most sensitive samples per class
            tmp_set = set(target.data.cpu().numpy())

            if not all([x in class_set for x in tmp_set]):
                print('Now visualizing token depth for class {}/1000.'.format(target[key].data.item()))
                max_std = 0

            class_set = class_set | tmp_set

            if value >= max_std:

                max_std=value
                idx=key

                file_path = "./token_act_visualization/"
                if not os.path.exists(file_path):
                    os.makedirs(file_path)

                target_token = cnt_token[idx,1:]
                array = np.reshape(target_token, (14, 14))

                plt.imshow(array, cmap='hot', interpolation='nearest')
                plt.axis('off')
                cb=plt.colorbar(shrink=0.8)

                if 1:
                    # save token depth heat map
                    plt.savefig(file_path + 'class{}_token_depth.jpg'.format(target[idx].data.item()))
                if 1:
                    # save original image
                    vutils.save_image(images[idx].data, file_path + 'class{}_ref.jpg'.format(target[idx].data.item()),
                                          normalize=True, scale_each=True)
                if 1:
                    # save concatenated image
                    # note that this snippet is not fully optimized in speed
                    im1 = cv2.imread(file_path + 'class{}_ref.jpg'.format(target[idx].data.item()))
                    im2 = cv2.imread(file_path + 'class{}_token_depth.jpg'.format(target[idx].data.item()))

                    if im1 is not None and im2 is not None:
                        cv2.imwrite(file_path + 'class{}_combined.jpg'.format(target[idx].data.item()), merge_image(im1, im2))

                cb.remove()

    print('Visualization done.')
    #exit()

    return
