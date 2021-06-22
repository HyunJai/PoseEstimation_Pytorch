# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import  # 현재 interpreter와 호환되지않는 새로운 특징을 사용하고자할 때 사용하는 pseudo-module
from __future__ import division
from __future__ import print_function   # ex) Python 2 -> Python 3 가능

import os
import yaml

import numpy as np
from easydict import EasyDict as edict
from datetime import datetime

log_name = datetime.now().strftime("%H:%M:%S__%Y-%m-%d")
config = edict()    # EasyDict allows to access dict values as attributes (works recursively)

config.OUTPUT_DIR=''
config.LOG_DIR = f'./loggers/{log_name}.log'
config.DATA_DIR = './data/'
config.GPUS = '0'
config.WORKERS = 4
config.PRINT_FREQ = 20

# Cudnn 관련 파라미터
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False  # 연산 처리속도가 감소되는 문제가 발생할 수 있음
config.CUDNN.ENABLED = True

# pose_resnet 관련 파라미터
POSE_RESNET = edict()
POSE_RESNET.NUM_LAYERS = 50
POSE_RESNET.DECONV_WITH_BIAS = False
POSE_RESNET.NUM_DECONV_LAYERS = 3
POSE_RESNET.NUM_DECONV_FILTERS = [256,256,256]
POSE_RESNET.NUM_DECONV_KERNELS = [4,4,4]
POSE_RESNET.FINAL_CONV_KERNELS = 1

MODEL_EXTRAS = {
    'pose_resnet': POSE_RESNET,
}

# NETWORK 관련 파라미터
config.MODEL = edict()
config.MODEL.CHECKPOINT = ''
config.MODEL.NAME = 'pose_resnet'
config.MODEL.INIT_WEIGHTS = True
config.MODEL.PRETRAINED = ''
config.MODEL.NUM_JOINTS = 17
config.MODEL.IMAGE_SIZE = [256,192]
config.MODEL.OUTPUT_SIZE = [config.MODEL.IMAGE_SIZE[0]//4, config.MODEL.IMAGE_SIZE[1]//4]
if config.MODEL.OUTPUT_SIZE[0] == 64:
    config.MODEL.INPUT_SIGMA = 7.0
elif config.MODEL.OUTPUT_SIZE[0] == 96:
    config.MODEL.INPUT_SIGMA = 9.0

config.MODEL.EXTRA = MODEL_EXTRAS[config.MODEL.NAME]

config.MODEL.STYLE = 'pytorch'

config.LOSS = edict()
config.LOSS.USE_TARGET_WEIGHT = True

# DATASET 관련 파라미터
config.DATASET = edict()
config.DATASET.ROOT = './data'
config.DATASET.DATASET = 'coco'
config.DATASET.TRAIN_SET = 'train'
config.DATASET.TEST_SET = 'valid'
config.DATASET.DATA_FORMAT = 'file'
config.DATASET.HYBRID_JOINTS_TYPE = ''
config.DATASET.SELECT_DATA = False

# Training Data Augmentation
config.DATASET.FLIP = True
config.DATASET.SCALE_FACTOR = 0.3
config.DATASET.ROT_FACTOR = 40