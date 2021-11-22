"""
CoaT architecture.

Paper: Co-Scale Conv-Attentional Image Transformers - https://arxiv.org/abs/2104.06399

Official CoaT code at: https://github.com/mlpc-ucsd/CoaT

Modified from timm/models/vision_transformer.py
"""
from copy import deepcopy
from functools import partial
from typing import Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

__all__ = [
    "coat_lite_tiny",
    "coat_lite_mini",
    "coat_lite_small"
]


def _cfg_coat(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed1.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'coat_tiny': _cfg_coat(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-coat-weights/coat_tiny-473c2a20.pth'
    ),
    'coat_mini': _cfg_coat(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-coat-weights/coat_mini-2c6baf49.pth'
    ),
    'coat_lite_tiny': _cfg_coat(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-coat-weights/coat_lite_tiny-461b07a7.pth'
    ),
    'coat_lite_mini': _cfg_coat(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-coat-weights/coat_lite_mini-d7842000.pth'
    ),
    'coat_lite_small': _cfg_coat(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-coat-weights/coat_lite_small-fea1d5a1.pth'
    ),
}


def masked_sin_pos_encoding(x, mask, num_pos_feats, temperature=10000, scale=2 * math.pi):
    """ Masked Sinusoidal Positional Encoding

    Parameters:
        x: [PATCH] tokens
        mask: the padding mask for [PATCH] tokens
        num_pos_feats: the size of channel dimension
        temperature: the temperature value
        scale: the normalization scale

    Returns:
        pos: Sinusoidal positional encodings
    """

    num_pos_feats = num_pos_feats // 2
    not_mask = ~mask

    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)

    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t

    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3)

    return pos


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvRelPosEnc(nn.Module):
    """ Convolutional relative position encoding. """

    def __init__(self, Ch, h, window):
        """
        Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding. It can have two forms:
                1. An integer of window size, which assigns all attention heads with the same window s
                    size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits (
                    e.g. {window size 1: #attention head split 1, window size 2: #attention head split 2})
                    It will apply different window size to the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1
            # Determine padding size.
            # Ref: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2d(cur_head_split * Ch, cur_head_split * Ch,
                                 kernel_size=(cur_window, cur_window),
                                 padding=(padding_size, padding_size),
                                 dilation=(dilation, dilation),
                                 groups=cur_head_split * Ch,
                                 )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size: Tuple[int, int]):
        B, h, N, Ch = q.shape
        H, W = size
        assert N == 1 + H * W

        # Convolutional relative position encoding.
        q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
        v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]

        v_img = v_img.transpose(-1, -2).reshape(B, h * Ch, H, W)
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
        conv_v_img_list = []
        for i, conv in enumerate(self.conv_list):
            conv_v_img_list.append(conv(v_img_list[i]))
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = conv_v_img.reshape(B, h, Ch, H * W).transpose(-1, -2)

        EV_hat = q_img * conv_v_img
        EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
        return EV_hat


class RonfiguredAttentionModule(nn.Module):
    """ Factorized attention with convolutional relative position encoding class. """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., shared_crpe=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size: Tuple[int, int], det=None, cross_attn=False, cross_attn_mask=None):

        # projection before projectioning
        if not cross_attn:
            B, N, C = x.shape
            x = torch.cat([x, det], dim=1)
            full_qkv = self.qkv(x)
            patch_qkv, det_qkv = full_qkv[:, :N, :], full_qkv[:, N:, :]
        else:
            B, N, C = x[0].shape
            _, ori_H, ori_W, _ = x[1].shape
            ori_N = ori_H * ori_W

            shifted_x = x[0]
            cross_x = x[1].view(B, ori_N, C)
            x = torch.cat([shifted_x, cross_x, det], dim=1)
            full_qkv = self.qkv(x)
            patch_qkv, cross_patch_qkv, det_qkv = \
                full_qkv[:, :N, :], full_qkv[:, N:N + ori_N, :], full_qkv[:, N + ori_N:, :]

        # [PATCH x PATCH] self-attention
        # Generate Q, K, V.
        patch_qkv = patch_qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        patch_q, patch_k, patch_v = patch_qkv[0], patch_qkv[1], patch_qkv[2]  # [B, h, N, Ch]

        # Factorized attention.
        k_softmax = patch_k.softmax(dim=2)
        factor_att = k_softmax.transpose(-1, -2) @ patch_v
        factor_att = patch_q @ factor_att

        # Convolutional relative position encoding.
        crpe = self.crpe(patch_q, patch_v, size=size)  # [B, h, N, Ch]

        # Merge and reshape.
        patch_x = self.scale * factor_att + crpe
        patch_x = patch_x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]

        # extract qkv for [DET] tokens
        det_qkv = det_qkv.view(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        det_q, det_k, det_v = det_qkv[0], det_qkv[1], det_qkv[2]

        # if cross-attention is activated
        if cross_attn:

            # reconstruct the spatial form of [PATCH] tokens for global [DET x PATCH] attention
            cross_patch_qkv = cross_patch_qkv.view(B, ori_H, ori_W, 3, self.num_heads, C // self.num_heads)
            patch_kv = cross_patch_qkv[:, :, :, 1:, :, :].permute(3, 0, 4, 1, 2, 5).contiguous()
            patch_kv = patch_kv.view(2, B, self.num_heads, ori_H * ori_W, -1)

            # extract "key and value" of [PATCH] tokens for cross-attention
            cross_patch_k, cross_patch_v = patch_kv[0], patch_kv[1]

            # bind key and value of [PATCH] and [DET] tokens for [DET X [PATCH, DET]] attention
            det_k, det_v = torch.cat([cross_patch_k, det_k], dim=2), torch.cat([cross_patch_v, det_v], dim=2)

        # [DET x DET] self-attention or binded [DET x [PATCH, DET]] attention
        det_q = det_q * self.scale
        det_attn = (det_q @ det_k.transpose(-2, -1))
        # apply cross-attention mask if available
        if cross_attn_mask is not None:
            det_attn = det_attn + cross_attn_mask

        det_attn = det_attn.softmax(dim=-1)
        det_x = (det_attn @ det_v).transpose(1, 2).reshape(B, -1, C)

        # projection for outputs from multi-head
        x = torch.cat([patch_x.view(B, N, C), det_x], dim=1)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        # decompose after FFN into [PATCH] and [DET] tokens
        patch_x = x[:, :N, :]
        det_x = x[:, N:, :]

        return patch_x, det_x


class ConvPosEnc(nn.Module):
    """ Convolutional Position Encoding.
        Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        assert N == 1 + H * W

        # Extract CLS token and image tokens.
        cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]

        # Depthwise convolution.
        feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)

        # Combine with CLS token.
        x = torch.cat((cls_token, x), dim=1)

        return x


class SerialBlock(nn.Module):
    """ Serial block class.
        Note: In this implementation, each serial block only contains a conv-attention and a FFN (MLP) module. """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, shared_cpe=None, shared_crpe=None):
        super().__init__()

        # Conv-Attention.
        self.cpe = shared_cpe

        self.norm1 = norm_layer(dim)
        self.factoratt_crpe = RonfiguredAttentionModule(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP.
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, size: Tuple[int, int],
                # additional inputs for RAM
                pos, cross_attn, cross_attn_mask):

        B, L, C = x.shape
        shortcut = x

        H, W = size
        x = self.norm1(x)
        x, det = x[:, :-self.det_token_num, :], x[:, -self.det_token_num:, :]
        orig_x = x[:, 1:, :].view(B, H, W, C)

        # projects det positional encoding: make the channel size suitable for the current layer
        patch_pos, det_pos = pos
        det_pos = self.det_pos_linear(det_pos)

        # prepare cross-attn and add positional encodings
        if cross_attn:
            # patch token (for cross-attention) + Sinusoidal pos encoding
            cross_x = orig_x + patch_pos
            # det token + learnable pos encoding
            det = det + det_pos
            x = (self.cpe(x, size), cross_x) # (x, cross_x)
        else:
            # it cross_attn is decativated, only [PATCH] and [DET] self-attention are performed
            det = det + det_pos
            x = self.cpe(x, size)

        # Reconfigured Conv-Attention (RAM)
        x, det = self.factoratt_crpe(x, size,
                                # additional parameters
                                det=det,
                                cross_attn=cross_attn,
                                cross_attn_mask=cross_attn_mask)

        x = torch.cat([x, det], dim=1)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Parameters:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""

        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class CoaT(nn.Module):
    """ CoaT class. """

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=(0, 0, 0, 0),
        serial_depths=(0, 0, 0, 0), parallel_depth=0, num_heads=0, mlp_ratios=(0, 0, 0, 0), qkv_bias=True,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
        return_interm_layers=True, out_features=None, crpe_window=None, **kwargs):
        super().__init__()
        crpe_window = crpe_window or {3: 2, 5: 3, 7: 3}
        self.return_interm_layers = return_interm_layers
        self.out_features = out_features
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]
        self.num_classes = num_classes

        # Patch embeddings.
        img_size = to_2tuple(img_size)
        self.patch_embed1 = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dims[0], norm_layer=nn.LayerNorm)
        self.patch_embed2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0],
            embed_dim=embed_dims[1], norm_layer=nn.LayerNorm)
        self.patch_embed3 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1],
            embed_dim=embed_dims[2], norm_layer=nn.LayerNorm)
        self.patch_embed4 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[2],
            embed_dim=embed_dims[3], norm_layer=nn.LayerNorm)

        # Class tokens.
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dims[2]))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # Convolutional position encodings.
        self.cpe1 = ConvPosEnc(dim=embed_dims[0], k=3)
        self.cpe2 = ConvPosEnc(dim=embed_dims[1], k=3)
        self.cpe3 = ConvPosEnc(dim=embed_dims[2], k=3)
        self.cpe4 = ConvPosEnc(dim=embed_dims[3], k=3)

        # Convolutional relative position encodings.
        self.crpe1 = ConvRelPosEnc(Ch=embed_dims[0] // num_heads, h=num_heads, window=crpe_window)
        self.crpe2 = ConvRelPosEnc(Ch=embed_dims[1] // num_heads, h=num_heads, window=crpe_window)
        self.crpe3 = ConvRelPosEnc(Ch=embed_dims[2] // num_heads, h=num_heads, window=crpe_window)
        self.crpe4 = ConvRelPosEnc(Ch=embed_dims[3] // num_heads, h=num_heads, window=crpe_window)

        # Disable stochastic depth.
        dpr = drop_path_rate
        assert dpr == 0.0

        # Serial blocks 1.
        self.serial_blocks1 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[0], num_heads=num_heads, mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer,
                shared_cpe=self.cpe1, shared_crpe=self.crpe1
            )
            for _ in range(serial_depths[0])]
        )

        # Serial blocks 2.
        self.serial_blocks2 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[1], num_heads=num_heads, mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer,
                shared_cpe=self.cpe2, shared_crpe=self.crpe2
            )
            for _ in range(serial_depths[1])]
        )

        # Serial blocks 3.
        self.serial_blocks3 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[2], num_heads=num_heads, mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer,
                shared_cpe=self.cpe3, shared_crpe=self.crpe3
            )
            for _ in range(serial_depths[2])]
        )

        # Serial blocks 4.
        self.serial_blocks4 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[3], num_heads=num_heads, mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer,
                shared_cpe=self.cpe4, shared_crpe=self.crpe4
            )
            for _ in range(serial_depths[3])]
        )

        # Classification head(s).
        self.norm2 = self.norm3 =self.norm4 = None
        if not self.return_interm_layers:
            self.norm2 = self.norm3 = None
            self.norm4 = norm_layer(embed_dims[3])

            # CoaT-Lite series: Use feature of last scale for classification.
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights.
        trunc_normal_(self.cls_token1, std=.02)
        trunc_normal_(self.cls_token2, std=.02)
        trunc_normal_(self.cls_token3, std=.02)
        trunc_normal_(self.cls_token4, std=.02)
        self.apply(self._init_weights)

        # dict to access
        self.stages = [self.serial_blocks1, self.serial_blocks2, self.serial_blocks3, self.serial_blocks4]
        self.patch_embeds = [self.patch_embed1, self.patch_embed2, self.patch_embed3, self.patch_embed4]
        self.cls_tokens = [self.cls_token1, self.cls_token2, self.cls_token3, self.cls_token4]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token1', 'cls_token2', 'cls_token3', 'cls_token4', 'det_pos_embed', 'det_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def insert_cls(self, x, cls_token):
        """ Insert CLS token. """
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x

    def remove_cls(self, x):
        """ Remove CLS token. """
        return x[:, 1:, :]

    # added for ViDT with Coat
    def finetune_det(self, method, det_token_num=100, pos_dim=256, cross_indices=[3]):
        """ A funtion to add neccessary (leanable) variables to Swin Transformer for object detection

            Parameters:
                det_token_num: the number of object to detect, i.e., number of object queries
                pos_dim: the channel dimension of positional encodings for [DET] and [PATCH] tokens
                cross_indices: the indices where to use the [DET X PATCH] cross-attention
                    there are four possible stages in [0, 1, 2, 3]. 3 indicates Stage 4 in the ViDT paper.
        """

        # which method?
        self.method = method

        # how many object we detect?
        self.det_token_num = det_token_num
        self.det_token = nn.Parameter(torch.zeros(1, det_token_num, self.embed_dims[0]))
        self.det_token = trunc_normal_(self.det_token, std=.02)

        # dim size of pos encoding
        self.pos_dim = pos_dim

        # learnable positional encoding for detection tokens
        det_pos_embed = torch.zeros(1, det_token_num, pos_dim)
        det_pos_embed = trunc_normal_(det_pos_embed, std=.02)
        self.det_pos_embed = torch.nn.Parameter(det_pos_embed)

        # info for detection
        self.num_channels = [self.embed_dims[i+1] for i in range(len(self.embed_dims)-1)]
        if method == 'vidt':
            self.num_channels.append(self.pos_dim) # default: 256 (same to the default pos_dim)
        self.cross_indices = cross_indices
        # divisor to reduce the spatial size of the mask
        self.mask_divisor = 2 ** (len(self.embed_dims) - len(self.cross_indices))

        # projection matrix for det pos encoding in each Swin layer (there are 4 blocks)
        for stage_id, stage in enumerate(self.stages):
            for block in stage:
                block.det_token_num = det_token_num
                block.det_pos_linear = nn.Linear(pos_dim, self.embed_dims[stage_id])

            # det channel expansion
            if stage_id > 0:
                det_exp = nn.Linear(self.embed_dims[stage_id-1], self.embed_dims[stage_id], bias=False)
                trunc_normal_(det_exp.weight, std=.02)
                det_exp_name = f'det_exp_{stage_id}'
                self.add_module(det_exp_name, det_exp)
                det_exp_norm = nn.LayerNorm(self.embed_dims[stage_id])
                nn.init.constant_(det_exp_norm.bias, 0)
                nn.init.constant_(det_exp_norm.weight, 1.0)
                det_exp_norm_name = f'det_exp_norm_{stage_id}'
                self.add_module(det_exp_norm_name, det_exp_norm)

        self.det_exps = [self.det_exp_1, self.det_exp_2, self.det_exp_3]
        self.det_exp_norms = [self.det_exp_norm_1, self.det_exp_norm_2, self.det_exp_norm_3]

        # neck-free model do not require downsamling at the last stage
        if method == 'vidt':
            self.patch_embed5 = PatchEmbed(
            patch_size=2, in_chans=self.embed_dims[-1],
            embed_dim=pos_dim, norm_layer=nn.LayerNorm)
            self.patch_embeds.append(self.patch_embed5)

    def forward_stage(self, x, H, W, stage_fn, det_pos, input_mask, cross_attn, dim):
        B = x.shape[0]

        # compute sinusoidal pos encoding and cross-attn mask here to avoid redundant computation
        if cross_attn:

            _H, _W = input_mask.shape[1:]
            if not (_H == H and _W == W):
                input_mask = F.interpolate(input_mask[None].float(), size=(H, W)).to(torch.bool)[0]

            # sinusoidal pos encoding for [PATCH] tokens used in cross-attention
            patch_pos = masked_sin_pos_encoding(x, input_mask, dim)

            # attention padding mask due to the zero padding in inputs
            # the zero (padded) area is masked by 1.0 in 'input_mask'
            cross_attn_mask = input_mask.float()
            cross_attn_mask = cross_attn_mask.masked_fill(cross_attn_mask != 0.0, float(-100.0)). \
                masked_fill(cross_attn_mask == 0.0, float(0.0))

            # pad for detection token (this padding is required to process the binded [PATCH, DET] attention
            cross_attn_mask = cross_attn_mask.view(B, H * W).unsqueeze(1).unsqueeze(2)
            cross_attn_mask = F.pad(cross_attn_mask, (0, self.det_token_num), value=0)

        else:
            patch_pos = None
            cross_attn_mask = None

        # zip pos encodings
        pos = (patch_pos, det_pos)

        for blk in stage_fn:

            # for selective cross-attention
            if cross_attn:
                _cross_attn = True
                _cross_attn_mask = cross_attn_mask
                _pos = pos # i.e., (patch_pos, det_pos)
            else:
                _cross_attn = False
                _cross_attn_mask = None
                _pos = (None, det_pos)

            # attention operations with RAM
            x = blk(x, size=(H, W),
                    # additional inputs
                    pos=_pos,
                    cross_attn=_cross_attn,
                    cross_attn_mask=_cross_attn_mask)

        x, det = x[:, :H * W + 1, :], x[:, H * W + 1:, :]
        x = self.remove_cls(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x, det, H, W

    def forward(self, x, mask):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        # expand det_token for all examples in the batch
        det_token = self.det_token.expand(B, -1, -1)

        # det pos encoding -> will be projected in each block
        det_pos = self.det_pos_embed

        # prepare a mask for cross attention
        mask = F.interpolate(mask[None].float(),
                     size=(H // self.mask_divisor, W // self.mask_divisor)).to(torch.bool)[0]

        # multi-scale [PATCH] tokens
        patch_outs = []
        for stage in range(len(self.embed_dims)):

            # whether to use cross-attention
            cross_attn = True if stage in self.cross_indices else False

            x = self.patch_embeds[stage](x)
            H, W = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.insert_cls(x, self.cls_tokens[stage])

            # merge with det token after det token expansion
            if stage > 0:
                det_token = self.det_exps[stage-1](det_token)
                det_token = self.det_exp_norms[stage-1](det_token)
            x = torch.cat([x, det_token], dim=1)

            x, det_token, H, W = self.forward_stage(x, H, W,
                                         self.stages[stage],
                                         # additional input for VIDT
                                         input_mask=mask,
                                         det_pos=det_pos,
                                         cross_attn=cross_attn,
                                         dim=self.embed_dims[stage])

            if stage > 0:
                patch_outs.append(x)

        if self.method == 'vidt':
            patch_outs.append(self.patch_embeds[-1](x))

        det_tgt = det_token.permute(0, 2, 1)
        det_pos = det_pos.permute(0, 2, 1)

        return patch_outs, det_tgt, det_pos


def checkpoint_filter_fn(state_dict):
    out_dict = {}
    for k, v in state_dict.items():
        # original model had unused norm layers, removing them requires filtering pretrained checkpoints

        # for ViDT
        if k.startswith('norm') or k.startswith('head'):
            continue

        out_dict[k] = v
    return out_dict


def _create_coat(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        CoaT, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model

@register_model
def coat_lite_tiny(pretrained=None, **kwargs):
    model_cfg = dict(
        patch_size=4, embed_dims=[64, 128, 256, 320], serial_depths=[2, 2, 2, 2], parallel_depth=0,
        num_heads=8, mlp_ratios=[8, 8, 4, 4], **kwargs)

    pretrained = True if pretrained == 'imagenet' else False
    model = _create_coat('coat_lite_tiny', pretrained=pretrained, **model_cfg)

    return model, 320

@register_model
def coat_lite_mini(pretrained=None, **kwargs):
    model_cfg = dict(
        patch_size=4, embed_dims=[64, 128, 320, 512], serial_depths=[2, 2, 2, 2], parallel_depth=0,
        num_heads=8, mlp_ratios=[8, 8, 4, 4], **kwargs)

    pretrained = True if pretrained == 'imagenet' else False
    model = _create_coat('coat_lite_mini', pretrained=pretrained, **model_cfg)

    return model, 512

@register_model
def coat_lite_small(pretrained=None, **kwargs):
    model_cfg = dict(
        patch_size=4, embed_dims=[64, 128, 320, 512], serial_depths=[3, 4, 6, 3], parallel_depth=0,
        num_heads=8, mlp_ratios=[8, 8, 4, 4], **kwargs)

    pretrained = True if pretrained == 'imagenet' else False
    model = _create_coat('coat_lite_small', pretrained=pretrained, **model_cfg)

    return model, 512

