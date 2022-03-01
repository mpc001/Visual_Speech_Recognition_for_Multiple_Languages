#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import os
import cv2
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save2npz(filename, data=None):
    """save2npz.

    :param filename: str, the filename to save the numpy.ndarray.
    :param data: numpy.ndarray, the data to be saved.
    """
    assert data is not None, "data is {}".format(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(filename, data=data)


def save2avi(filename, data=None, fps=25):
    """save2avi.

    :param filename: str, the filename to save the video (.avi).
    :param data: numpy.ndarray, the data to be saved.
    :param fps: the chosen frames per second.
    """
    assert data is not None, "data is {}".format(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc("F", "F", "V", "1")
    writer = cv2.VideoWriter(filename, fourcc, fps, (data[0].shape[1], data[0].shape[0]), 0)
    for frame in data:
        writer.write(frame)
    writer.release()
