#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import random
import torch
import cv2
import numpy as np
from scipy import signal


__all__ = [
    "Compose", "Normalize", "CenterCrop", "AddNoise", "NormalizeUtterance", \
    "Identity", "SpeedRate", "ExpandDims"
]


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, img):
        for t in self.preprocess:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return (img - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)


class NormalizeUtterance():
    """Normalize per raw audio by removing the mean and divided by the standard deviation
    """
    def __call__(self, signal):
        signal_std = 0. if np.std(signal)==0. else np.std(signal)
        signal_mean = np.mean(signal)
        return (signal - signal_mean) / signal_std



class CenterCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        frames, h, w = img.shape
        th, tw = self.crop_size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        return img[:, delta_h:delta_h+th, delta_w:delta_w+tw]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.crop_size)


class AddNoise(object):
    """Add SNR noise [-1, 1]
    """

    def __init__(self, noise, snr_target=None, snr_levels=[-5, 0, 5, 10, 15, 20, 9999]):
        assert noise.dtype in [np.float32, np.float64], "noise only supports float data type"
        self.noise = noise
        self.snr_levels = snr_levels
        self.snr_target = snr_target

    def get_power(self, clip):
        clip2 = clip.copy()
        clip2 = clip2 **2
        return np.sum(clip2) / (len(clip2) * 1.0)

    def __call__(self, signal):
        assert signal.dtype in [np.float32, np.float64], "signal only supports float32 data type"
        snr_target = random.choice(self.snr_levels) if not self.snr_target else self.snr_target
        if snr_target == 9999:
            return signal
        else:
            # -- get noise
            start_idx = random.randint(0, len(self.noise)-len(signal))
            noise_clip = self.noise[start_idx:start_idx+len(signal)]

            sig_power = self.get_power(signal)
            noise_clip_power = self.get_power(noise_clip)
            factor = (sig_power / noise_clip_power ) / (10**(snr_target / 10.0))
            desired_signal = (signal + noise_clip*np.sqrt(factor)).astype(np.float32)
            return desired_signal


class Identity(object):
    """Identity
    """
    def __init__(self, ):
        pass
    def __call__(self, array):
        return array


class SpeedRate(object):
    """Subsample/Upsample the number of frames in a sequence.
    """

    def __init__(self, speed_rate=1.0):
        """__init__.

        :param speed_rate: float, the speed rate between the frame rate of \
            the input video and the frame rate used for training.
        """
        self._speed_rate = speed_rate

    def __call__(self, x):
        """
        Args:
            img (numpy.ndarray): sequence to be sampled.
        Returns:
            numpy.ndarray: sampled sequence.
        """
        if self._speed_rate <= 0:
            raise ValueError("speed_rate should be greater than zero.")
        if self._speed_rate == 1.:
            return x
        old_length = x.shape[0]
        new_length = int(old_length / self._speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length, endpoint=False)
        new_indices = list(map(int, new_indices))
        x = x[new_indices]
        return x

class ExpandDims(object):
    """ExpandDims."""


    def __init__(self,):
        """__init__."""

    def __call__(self, x):
        """__call__.

        :param x: numpy.ndarray, Expand the shape of an array.
        """
        return np.expand_dims(x, axis=1) if x.ndim == 1 else x
