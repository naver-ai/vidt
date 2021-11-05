# ------------------------------------------------------------------------
# DETR
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Additionally modified by NAVER Corp. for ViDT
# ------------------------------------------------------------------------
"""Build a VIDT (without Neck) detector for object detection."""

import torch
import torch.nn as nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import torch.nn.functional as F
from methods.swin_w_ram import swin_nano, swin_tiny, swin_small, swin_base_win7, swin_large_win7
from methods.coat_w_ram import coat_lite_tiny, coat_lite_mini, coat_lite_small
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessor import PostProcess


class Detector(nn.Module):
    """ This is a neck-free detector using "Swin with RAM" """

    def __init__(self, backbone, reduced_dim, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            reduced_dim: the channel dim for the detection head
            num_classes: number of object classes
        """

        super().__init__()
        # import pdb;pdb.set_trace()
        self.backbone = backbone
        hidden_dim = backbone.num_channels[-1]
        self.input_proj = nn.Sequential(
                nn.Conv2d(hidden_dim, reduced_dim, kernel_size=1),
                nn.GroupNorm(32, reduced_dim),)
        self.class_embed = MLP(reduced_dim, reduced_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(reduced_dim, reduced_dim, 4, 3)

        ## init
        nn.init.xavier_uniform_(self.input_proj[0].weight, gain=1)
        nn.init.constant_(self.input_proj[0].bias, 0)
        ##

    def forward(self, samples: NestedTensor):
        """ The forward step of ViDT

        Parameters:
            The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        Returns:
            A dictionary having the key and value pairs below:
            - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
        """

        # import pdb;pdb.set_trace()
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # return final [DET] tokens
        _, det, _ = self.backbone(samples.tensors, samples.mask)

        # projection
        x = self.input_proj(det.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)

        # predictions
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x).sigmoid()

        # final prediction is made the last decoding layer
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        return out

    def forward_return_attention(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        attention = self.backbone(samples.tensors, return_attention=True)
        return attention


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):

    if args.dataset_file == 'coco':
        num_classes = 91

    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    if args.backbone_name == 'swin_nano':
        backbone, hidden_dim = swin_nano(pretrained=args.pre_trained)
    elif args.backbone_name == 'swin_tiny':
        backbone, hidden_dim = swin_tiny(pretrained=args.pre_trained)
    elif args.backbone_name == 'swin_small':
        backbone, hidden_dim = swin_small(pretrained=args.pre_trained)
    elif args.backbone_name == 'swin_base_win7_22k':
        backbone, hidden_dim = swin_base_win7(pretrained=args.pre_trained)
    elif args.backbone_name == 'swin_large_win7_22k':
        backbone, hidden_dim = swin_large_win7(pretrained=args.pre_trained)
    elif args.backbone_name == 'coat_lite_tiny':
        backbone, hidden_dim = coat_lite_tiny(pretrained=args.pre_trained)
    elif args.backbone_name == 'coat_lite_mini':
        backbone, hidden_dim = coat_lite_mini(pretrained=args.pre_trained)
    elif args.backbone_name == 'coat_lite_small':
        backbone, hidden_dim = coat_lite_small(pretrained=args.pre_trained)
    else:
        raise ValueError(f'backbone {args.backbone_name} not supported')

    backbone.finetune_det(method=args.method,
                          det_token_num=args.det_token_num,
                          pos_dim=args.pos_dim,
                          cross_indices=args.cross_indices)

    # import pdb;pdb.set_trace()
    model = Detector(
        backbone,
        reduced_dim=args.reduced_dim,
        num_classes=num_classes,
    )
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    # TODO this is a hack
    # if args.aux_loss:
    #     aux_weight_dict = {}
    #     for i in range(args.dec_layers - 1):
    #         aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    #     weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors

