from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import torch
import torch.nn as nn
from collections import OrderedDict

from torch.nn.modules import padding

BN_MOMENTUM = 0.1   # Batch Normalization : 딥러닝 모델을 학습시킬때 사용되는 레이어중 하나로, 모델 학습시 수렴의 안전성과 속도 향상
                    # Momentum : 신경망의 학습 안전성과 속도를 높여 학습을 잘하기위해 사용됨. 모멘텀을 사용하면 가중치 값이 바로 바뀌지 않고
                    #            어느정도 일정한 방향을 유지하면서 움직임. 가속도처럼 같은 방향으로 더 많이 변화시켜 학습속도를 높여줘 빠른 학습을 진행함
logger = logging.getLogger(__name__)    # 여러 번 호출하면 같은 로거 객체에 대한 참조가 변환됨

def Conv33(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,  # Kernel 크기 설정
                        stride=stride,  # Filter 한번 이동 간격
                        padding=1, bias=False)  # Padding의 크기 만큼 이미지의 상하좌우에 '0'으로된 pad 둘러짐

class BasicB(nn.Module):    # Layer와 output 반환하는 Forward 메소드 포함
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicB, self).__init__()  # class 형태의 모델은 항상 nn.Module을 상속받아야하며,
                                        # suepr(모델명,self).__init__()을 통해 nn.Moudule.__init__()을 실행시키는 코드 필요
        self.conv1 = Conv33(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv33(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out