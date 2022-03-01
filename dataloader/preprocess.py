#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import cv2
import numpy as np                                                               
from skimage import transform as tf


def linear_interpolate(landmarks, start_idx, stop_idx):
    """linear_interpolate.

    :param landmarks: ndarray, input landmarks to be interpolated.
    :param start_idx: int, the start index for linear interpolation.
    :param stop_idx: int, the stop for linear interpolation.
    """
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks


def warp_img(src, dst, img, std_size):
    """warp_img.

    :param src: ndarray, source coordinates.
    :param dst: ndarray, destination coordinates. 
    :param img: ndarray, an input image.
    :param std_size: tuple (rows, cols), shape of the output image generated.
    """
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # wrap the frame image
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform


def apply_transform(transform, img, std_size):
    """apply_transform.

    :param transform: Transform object, containing the transformation parameters \
                      and providing access to forward and inverse transformation functions.
    :param img: ndarray, an input image.
    :param std_size: tuple (rows, cols), shape of the output image generated.
    """
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped


def cut_patch(img, landmarks, height, width, threshold=5):
    """cut_patch.

    :param img: ndarray, an input image.
    :param landmarks: ndarray, the corresponding landmarks for the input image.
    :param height: int, the distance from the centre to the side of of a bounding box.
    :param width: int, the distance from the centre to the side of of a bounding box.
    :param threshold: int, the threshold from the centre of a bounding box to the side of image.
    """
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:                                                
        center_y = height                                                    
    if center_y - height < 0 - threshold:                                    
        raise Exception('too much bias in height')                           
    if center_x - width < 0:                                                 
        center_x = width                                                     
    if center_x - width < 0 - threshold:                                     
        raise Exception('too much bias in width')                            
                                                                             
    if center_y + height > img.shape[0]:                                     
        center_y = img.shape[0] - height                                     
    if center_y + height > img.shape[0] + threshold:                         
        raise Exception('too much bias in height')                           
    if center_x + width > img.shape[1]:                                      
        center_x = img.shape[1] - width                                      
    if center_x + width > img.shape[1] + threshold:                          
        raise Exception('too much bias in width')                            
                                                                             
    cutted_img = np.copy(img[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img


def convert_bgr2gray(sequence):
    """convert_bgr2gray.

    :param sequence: ndarray, the RGB image sequence.
    """
    return np.stack([cv2.cvtColor(_, cv2.COLOR_BGR2GRAY) for _ in sequence], axis=0)
