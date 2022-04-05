'''
ViDT
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''

from methods.vidt.detector import build as vidt_build

def build_model(args, is_teacher=False):
    available_methods = ['vidt_wo_neck', 'vidt']

    if args.method not in available_methods:
        raise ValueError(f'method [{args.method}] is not supported')

    elif args.method == 'vidt':
        return vidt_build(args, is_teacher=is_teacher)
