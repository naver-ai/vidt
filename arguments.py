# ------------------------------------------------------------------------
# DETR
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Additionally modified by NAVER Corp. for ViDT
# ------------------------------------------------------------------------

import argparse

def str2bool(v, bool):

    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't'):
        return True
    elif v.lower() in ('false', 'f'):
        return False
    else:
        argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('Set ViDT', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--eval_size', default=800, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # * Learning rate schedule parameters
    parser.add_argument('--sched', default='warmupcos', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step", options:"step", "warmupcos"')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                         help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                         help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                         help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # * model setting
    parser.add_argument('--backbone_name', default='swin_tiny', type=str,
                        help="Name of the deit backbone to use")
    parser.add_argument('--pre_trained', default='imagenet', type=str,
                        help="set imagenet pretrained model path if not train yolos from scatch")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # * Dataset
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='/home/Research/MyData/COCO2017', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    # * Device and Log
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=False, type=lambda x: (str(x).lower() == 'true'), help='eval mode')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # * Training setup
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3457', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--distributed', action='store_true', help='')
    parser.add_argument('--num_workers', default=2, type=int)

    # * Pos encodig
    parser.add_argument('--position_embedding', default='sine', type=str)

    # * Transformer
    parser.add_argument('--pos_dim', default=256, type=int, help="Size of the embeeding for pos")
    parser.add_argument('--reduced_dim', default=256, type=int, help="Size of the embeddings for head")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int, # Deform-DETR: 1024, DETR: 2048
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # * Deformable Attention
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')


    ####### ViDT Params
    parser.add_argument('--method', default='vidt', type=str, help='method names in {vidt, vidt_wo_neck}')
    parser.add_argument("--det_token_num", default=100, type=int, help="Number of det token in the body backbone")
    parser.add_argument('--cross_indices', default=[3], nargs='+', type=int, help='stage ids for [DET x PATCH] cross-attention')

    # * Auxiliary Techniques
    parser.add_argument('--aux_loss', default=False, type=lambda x: (str(x).lower() == 'true'), help='auxiliary decoding loss')
    parser.add_argument('--with_box_refine', default=False, type=lambda x: (str(x).lower() == 'true'), help='iterative box refinement')

    # * Distillation with token matching
    parser.add_argument('--distil_loss_coef', default=4.0, type=float, help="Distillation coefficient")
    parser.add_argument('--distil_model', default=None, type=str, help="Distillation model in {vidt_tiny, vidt_small, vidt-base}")
    parser.add_argument('--distil_model_path', default=None, type=str, help="Distillation model path to load")
    #######

    ####### See DDETR: https://openreview.net/forum?id=LhbD74dsZFL for below techniques
    # cross-scale fusion
    parser.add_argument('--cross_scale_fusion', default=False, type=lambda x: (str(x).lower() == 'true'), help='use of scale fusion')
    # iou-aware
    parser.add_argument('--iou_aware', default=False, type=lambda x: (str(x).lower() == 'true'), help='use of iou-aware loss')
    parser.add_argument('--iouaware_loss_coef', default=2, type=float)
    # token label
    parser.add_argument('--token_label', default=False, type=lambda x: (str(x).lower() == 'true'), help='use of token label loss')
    parser.add_argument('--token_loss_coef', default=2, type=float)
    #######

    # * Logs
    parser.add_argument('--n_iter_to_acc', default=1, type=int, help='gradient accumulation step size')
    parser.add_argument('--print_freq', default=500, type=int, help='number of iteration to print training logs')

    return parser