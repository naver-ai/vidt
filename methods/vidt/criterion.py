# ------------------------------------------------------------------------
# DETR
# Copyright (c) 2020 Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Additionally Modified by Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Additionally modified by NAVER Corp. for ViDT
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from methods.segmentation import (dice_loss, sigmoid_focal_loss)
import copy


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """

        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """

        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_iouaware(self, outputs, targets, indices, num_boxes):
        assert 'pred_ious' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_ious = outputs['pred_ious'][idx]  # logits
        src_ious = src_ious.squeeze(1)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        iou = torch.diag(box_ops.box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))[0])

        losses = {}
        loss_iouaware = F.binary_cross_entropy_with_logits(src_ious, iou, reduction='none')
        losses['loss_iouaware'] = loss_iouaware.sum() / num_boxes
        return losses

    def loss_tokens(self, outputs, targets, num_boxes):
        enc_token_class_unflat = outputs['pred_logits']

        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()

        bs, n, h, w = target_masks.shape
        mask = torch.zeros((bs, h, w), dtype=torch.bool, device=target_masks.device)
        for j in range(n):
            target_masks[:, j] &= target_masks[:, j] ^ mask
            mask |= target_masks[:, j]
        target_classes_pad = torch.stack([F.pad(t['labels'], (0, n - len(t['labels']))) for t in targets])
        final_mask = torch.sum(target_masks * target_classes_pad[:, :, None, None], dim=1)  # (bs, h, w)
        final_mask_onehot = torch.zeros((bs, h, w, self.num_classes), dtype=torch.float32, device=target_masks.device)
        final_mask_onehot.scatter_(-1, final_mask.unsqueeze(-1), 1)  # (bs, h, w, 91)

        final_mask_onehot[..., 0] = 1 - final_mask_onehot[..., 0]  # change index 0 from background to foreground

        loss_token_focal = 0
        loss_token_dice = 0
        for i, enc_token_class in enumerate(enc_token_class_unflat):
            _, h, w, _ = enc_token_class.shape

            final_mask_soft = F.adaptive_avg_pool2d(final_mask_onehot.permute(0, 3, 1, 2), (h,w)).permute(0, 2, 3, 1)

            enc_token_class = enc_token_class.flatten(1, 2)
            final_mask_soft = final_mask_soft.flatten(1, 2)
            loss_token_focal += sigmoid_focal_loss(enc_token_class, final_mask_soft, num_boxes)
            loss_token_dice += dice_loss(enc_token_class, final_mask_soft, num_boxes)

        losses = {
            'loss_token_focal': loss_token_focal,
            'loss_token_dice': loss_token_dice,
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'iouaware': self.loss_iouaware,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, distil_tokens=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        _indices = indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}

                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'enc_tokens' in outputs:
            l_dict = self.loss_tokens(outputs['enc_tokens'], targets, num_boxes)
            losses.update(l_dict)

        # distil. loss
        if distil_tokens is not None:
            patches, teacher_patches = distil_tokens[0]['patch_token'], distil_tokens[1]['patch_token']
            body_det, teacher_body_det = distil_tokens[0]['body_det_token'], distil_tokens[1]['body_det_token']
            neck_det, teacher_neck_det = distil_tokens[0]['neck_det_token'], distil_tokens[1]['neck_det_token']

            distil_loss = 0.0
            for patch, teacher_patch in zip(patches, teacher_patches):
                b, c, w, h = patch.shape
                patch = patch.permute(0, 2, 3, 1).contiguous().view(b*w*h, c)
                teacher_patch = teacher_patch.permute(0, 2, 3, 1).contiguous().view(b*w*h, c).detach()
                distil_loss += torch.mean(torch.sqrt(torch.sum(torch.pow(patch - teacher_patch, 2), dim=-1)))

            b, d, c = body_det.shape
            body_det = body_det.contiguous().view(b*d, c)
            teacher_body_det = teacher_body_det.contiguous().view(b*d, c).detach()
            distil_loss += torch.mean(torch.sqrt(torch.sum(torch.pow(body_det - teacher_body_det, 2), dim=-1)))

            l, b, d, c = neck_det.shape
            neck_det = neck_det.contiguous().view(l*b*d, c)
            teacher_neck_det = teacher_neck_det.contiguous().view(l*b*d, c).detach()
            distil_loss += (torch.mean(torch.sqrt(torch.sum(torch.pow(neck_det - teacher_neck_det, 2), dim=-1))) * l)

            l_dict = {'loss_distil': torch.sqrt(distil_loss)}
            losses.update(l_dict)

        return losses



