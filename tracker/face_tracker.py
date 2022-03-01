#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import collections
from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor
from .utils import get_landmarks
from .utils import extract_opencv_generator


class FaceTracker(object):
    """FaceTracker."""

    def __init__(self, device="cuda:0"):
        """__init__.

        :param device: str, contain the device on which a torch.Tensor is or will be allocated.
        """

        # Create a RetinaFace detector using Resnet50 backbone.
        self.face_detector = RetinaFacePredictor(
            device=device,
            threshold=0.8,
            model=RetinaFacePredictor.get_model('resnet50')
        )
        # Create FAN for alignmentm, default model is '2dfan2'
        alignment_weights = None
        self.landmark_detector = FANPredictor(device=device, model=alignment_weights)

    def tracker(self, filename):
        """tracker.

        :param filename: str, the filename for the video
        """

        face_info = collections.defaultdict(list)
        frame_gen = extract_opencv_generator(filename)

        while True:
            try:
                frame = frame_gen.__next__()
            except StopIteration:
                break
            # -- face detection
            detected_faces = self.face_detector(frame, rgb=False)
            # -- face alignment
            landmarks, scores = self.landmark_detector(frame, detected_faces, rgb=False)
            face_info['bbox'].append(detected_faces)
            face_info['landmarks'].append(landmarks)
            face_info['landmarks_scores'].append(scores)

        return get_landmarks(face_info)
