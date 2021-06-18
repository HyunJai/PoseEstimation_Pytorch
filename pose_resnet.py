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

class BottleN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes, kerenel_size=1, stride=stride,
                                bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(x)

        return x

class BottleN_CAFFE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleN_CAFFE, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kerne_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)   # inplace 연산은 결과값을 새로운 변수에 값을 저장하는 대신
                                            # 기존의 데이터를 대체하는것을 의미함
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out

class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):   # **변수 : key, value값으로 가져오는 'dictionary'로 처리함
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size= 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.makelayer(block, 64, layers[0])
        self.layer2 = self.makelayer(block, 128, layers[1], stride=2)
        self.layer3 = self.makelayer(block, 256, layers[2], stride=2)
        self.layer4 = self.makelayer(block, 512, layers[3], stride=2)

        self.deconv_layers = self.make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels = extra.NUM_DECONV_FILTERS[-1],
            out_channels = cfg.MODEL.NUM_JOINTS,
            kernel_size = extra.FINAL_CONV_KERNEL,
            stride = 1,
            padding = 1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def makelayer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, BN_MOMENTUM)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)   # *변수 : 함수 내부에서는 해당 변수를 'tuple'로 처리함

    def get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'Error: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'Error: num_deconv_layers is different len(num_deconv_kernels)'

        layers = []

        for i in range(num_layers):
            kernel, padding, output_padding = \
                self.get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d( # 여러 입력 평면으로 구성된 입력 이미지에 2D 전치 컨볼루션 연산자를 적용함
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kerne_size= kernel,
                    stride = 2,
                    padding = padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):   # 자료형 확인 구간
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)    # 정규분포N(mean, std^2)로부터의 값을 Tensor로 입력
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)    # value값을 Tensor값으로 넣어줌
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.wieght as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            logger.info('=> loading pretrained model {}'.format(pretrained))
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict): # 삽입된 순서를 기억하는 dictionary 자료형
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]

            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model 존재하지 않음')
            logger.error('=> 먼저 다운로드 받기')
            raise ValueError('imagenet pretrained model 존재하지 않음')

resnet_spec = {18: (BasicB, [2, 2, 2, 2]),
               34: (BasicB, [3, 4, 6, 3]),
               50: (BottleN, [3, 4, 6, 3]),
               101: (BottleN, [3, 4, 23, 3]),
               152: (BottleN, [3, 8, 36, 3])}

def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    style = cfg.MODEL.STYLE

    block_class, layers = resnet_spec[num_layers]

    if style == 'caffe':
        block_class = BottleN_CAFFE

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model