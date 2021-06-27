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

# 학습
config.TRAIN = edict()

config.TRAIN.SHORT_BOTTOM = True
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90,110]
config.TRAIN.LR = 0.001
config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140
config.TRAIN.RESUME = False

config.TRAIN.BATCH_SIZE = 64
config.TRAIN.SHUFFLE = True

# 테스트
config.TEST = edict()

config.TEST.BATCH_SIZE = 32
config.TEST.FLIP_TEST = True
config.TEST.POST_PROCESS = True
config.TEST.SHIFT_HEATMAP = True

config.TEST.USE_GT_BBOX = False
# NMS
config.TEST.OKS_THRE = 0.5
config.TEST.iN_VIS_THRE = 0.0
config.TEST.COCO_BBOX_FILE=''
config.TEST.MODEL_FILE = 1.0
config.TEST.IMAGE_TRE = 0.0
config.TEST.NMS_THRE = 1.0

# DEBUG
config.DEBUG = edict()
config.DEBUG.DEBUG = True
config.DEBUG.SAVE_BATCH_IMAGES_GT = False
config.DEBUG.SAVE_BATCH_IMAGES_PRED = False
config.DEBUG.SAVE_HEATMAPS_GT = False
config.DEBUG.SAVE_HEATMAPS_PRED = False

config.kps_suymmetry = [(1,2), (3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
config.kps_lines = [(0,1),(0,2),(6,8),(8,10),(5,7),(7,9),(12,14),(14,16),(11,13),(13,15),(5,6),(11,12)]
config.kps_sigmas = np.array([
    .26, .35, .25, 0, 0, .79, .79, .72, .72, .62, .62, 1.07, 1.07, 0.87, .87, .89, .89
]) / 10.0
config.ignore_kps = [3,4]
config.train_vis_dir = './train_visualize'
config.val_vis_dir = './valid_visualize'
config.pixel_means = np.array([[123.68, 116.78, 103.94]])
config.oks_nms = True
config.oks_nms_thr = 0.9
config.result_dir = './results'
