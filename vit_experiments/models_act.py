# --------------------------------------------------------
# The following code is based on:
## A-ViT
# https://github.com/NVlabs/A-ViT
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
# The code is modified to accomodate ViT training
# --------------------------------------------------------

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


### This is ViT-T (depth=12) model in our paper ###
@register_model
def tvit_tiny_patch8(pretrained=False, **kwargs):

    #from timm.models.res_vision_transformer import VisionTransformer
    from res_vision_transformer import VisionTransformer


    model = VisionTransformer(
        patch_size=8, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

### This is ViT-T+ (depth=18) model in our paper ###
@register_model
def tvit_small_patch8(pretrained=False, **kwargs):

    #from timm.models.res_vision_transformer import VisionTransformer
    from res_vision_transformer import VisionTransformer

    model = VisionTransformer(
        patch_size=8, embed_dim=192, depth=18, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


'''
@register_model
def vit_tiny_patch8(pretrained=False, **kwargs):

    from timm.models.vision_transformer import VisionTransformer

    model = VisionTransformer(
        patch_size=8, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
'''
