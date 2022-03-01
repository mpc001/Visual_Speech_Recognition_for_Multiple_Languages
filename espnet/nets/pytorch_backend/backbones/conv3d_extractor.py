#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from espnet.nets.pytorch_backend.backbones.modules.resnet import ResNet
from espnet.nets.pytorch_backend.backbones.modules.resnet import BasicBlock
from espnet.nets.pytorch_backend.backbones.modules.shufflenetv2 import ShuffleNetV2
from espnet.nets.pytorch_backend.transformer.convolution import Swish


# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)


class Conv3dResNet(torch.nn.Module):
    """Conv3dResNet module
    """

    def __init__(self, backbone_type="resnet", relu_type="swish"):
        """__init__.

        :param backbone_type: str, the type of a visual front-end.
        :param relu_type: str, activation function used in an audio front-end.
        """
        super(Conv3dResNet, self).__init__()

        self.backbone_type = backbone_type

        if self.backbone_type == "resnet":
            self.frontend_nout = 64
            self.trunk = ResNet(
                BasicBlock,
                [2, 2, 2, 2],
                relu_type=relu_type,
            )
        elif self.backbone_type == "shufflenet":
            shufflenet = ShuffleNetV2(
                input_size=96,
                width_mult=1.0
            )
            self.trunk = nn.Sequential(
                shufflenet.features,
                shufflenet.conv_last,
                shufflenet.globalpool,
            )
            self.frontend_nout = 24
            self.stage_out_channels = shufflenet.stage_out_channels[-1]

        # -- frontend3D
        if relu_type == 'relu':
            frontend_relu = nn.ReLU(True)
        elif relu_type == 'prelu':
            frontend_relu = nn.PReLU( self.frontend_nout )
        elif relu_type == 'swish':
            frontend_relu = Swish()

        self.frontend3D = nn.Sequential( 
            nn.Conv3d(
                in_channels=1,
                out_channels=self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            )
        )


    def forward(self, xs_pad):
        """forward.

        :param xs_pad: torch.Tensor, batch of padded input sequences.
        """
        # -- include Channel dimension
        xs_pad = xs_pad.unsqueeze(1)

        B, C, T, H, W = xs_pad.size()
        xs_pad = self.frontend3D(xs_pad)
        Tnew = xs_pad.shape[2]    # outpu should be B x C2 x Tnew x H x W
        xs_pad = threeD_to_2D_tensor( xs_pad )
        xs_pad = self.trunk(xs_pad)
        xs_pad = xs_pad.view(B, Tnew, xs_pad.size(1))

        return xs_pad
