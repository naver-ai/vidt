# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
from torch import nn
from util import box_ops
import torch.nn.functional as F


class PostProcess(nn.Module):
  """ This module converts the model's output into the format expected by the coco api"""

  def __init__(self, dataset_file):
    super().__init__()
    self.dataset_file = dataset_file

  @torch.no_grad()
  def forward(self, outputs, target_sizes, target_boxes=None):
    """ Perform the computation

    Parameters:
        outputs: raw outputs of the model
        target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                      For evaluation, this must be the original image size (before any data augmentation)
                      For visualization, this should be the image size after data augment, but before padding
    """

    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    if self.dataset_file == 'coco':
      prob = out_logits.sigmoid()
      topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
      scores = topk_values
      topk_boxes = topk_indexes // out_logits.shape[2]
      labels = topk_indexes % out_logits.shape[2]
      boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
      boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

      # and from relative [0, 1] to absolute [0, height] coordinates
      img_h, img_w = target_sizes.unbind(1)
      scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(torch.float32)
      boxes = boxes * scale_fct[:, None, :]

      results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

      return results

    elif self.dataset_file == 'voc':

      prob = F.softmax(out_logits, -1)
      scores, labels = prob.max(-1)

      # convert to [x0, y0, x1, y1] format
      boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
      # and from relative [0, 1] to absolute [0, height] coordinates
      img_h, img_w = target_sizes.unbind(1)
      scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(torch.float32)
      boxes = boxes * scale_fct[:, None, :]

      results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

      true_boxes = []
      for id in range(len(target_boxes)):
        true_box = box_ops.box_cxcywh_to_xyxy(target_boxes[id])
        scale_fct = torch.stack([img_w[id], img_h[id], img_w[id], img_h[id]], dim=0).to(torch.float32)
        true_boxes.append(true_box * scale_fct)

      return results, true_boxes

