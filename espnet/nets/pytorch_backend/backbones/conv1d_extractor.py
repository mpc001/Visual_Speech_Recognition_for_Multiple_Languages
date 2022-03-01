#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import logging
import torch
import numpy as np
from espnet.nets.pytorch_backend.backbones.modules.resnet1d import ResNet1D
from espnet.nets.pytorch_backend.backbones.modules.resnet1d import BasicBlock1D


class Conv1dResNet(torch.nn.Module):
    """Conv1dResNet
    """

    def __init__(self, relu_type="swish", a_upsample_ratio=1):
        """__init__.

        :param relu_type: str, Activation function used in an audio front-end.
        :param a_upsample_ratio: int, The ratio related to the \
            temporal resolution of output features of the frontend. \
            a_upsample_ratio=1 produce features with a fps of 25.
        """
        
        super(Conv1dResNet, self).__init__()
        self.a_upsample_ratio=a_upsample_ratio
        self.trunk = ResNet1D(
            BasicBlock1D,
            [2, 2, 2, 2],
            relu_type=relu_type,
            a_upsample_ratio=a_upsample_ratio
        )

    def forward(self, xs_pad):
        """forward.

        :param xs_pad: torch.Tensor, batch of padded input sequences (B, Tmax, idim)
        """
        B, T, C = xs_pad.size()
        xs_pad = xs_pad[:,:T//640*640,:]
        xs_pad = xs_pad.transpose(1,2)
        xs_pad = self.trunk(xs_pad)
        # -- from B x C x T to B x T x C
        xs_pad = xs_pad.transpose(1, 2)
        return xs_pad
