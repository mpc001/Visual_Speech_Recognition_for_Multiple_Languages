#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import cv2


def extract_opencv_generator(filename):
    """extract_opencv_generator.

    :param filename: str, the filename for video.
    """
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            yield frame
        else:
            break
    cap.release()


def get_landmarks(multi_sub_landmarks):
    """get_landmarks.

    :param multi_sub_landmarks: dict, a dictionary contains landmarks, bbox, and landmarks_scores.
    """

    landmarks = [None] * len( multi_sub_landmarks["landmarks"])
    for frame_idx in range(len(landmarks)):
        if len(multi_sub_landmarks["landmarks"][frame_idx]) == 0:
            continue
        else:
            # -- decide person id using maximal bounding box  0: Left, 1: top, 2: right, 3: bottom, probability
            max_bbox_person_id = 0
            max_bbox_len = multi_sub_landmarks["bbox"][frame_idx][max_bbox_person_id][2] + \
                           multi_sub_landmarks["bbox"][frame_idx][max_bbox_person_id][3] - \
                           multi_sub_landmarks["bbox"][frame_idx][max_bbox_person_id][0] - \
                           multi_sub_landmarks["bbox"][frame_idx][max_bbox_person_id][1]
            landmark_scores = multi_sub_landmarks["landmarks_scores"][frame_idx][max_bbox_person_id]
            for temp_person_id in range(1, len(multi_sub_landmarks["bbox"][frame_idx])):
                temp_bbox_len = multi_sub_landmarks["bbox"][frame_idx][temp_person_id][2] + \
                                multi_sub_landmarks["bbox"][frame_idx][temp_person_id][3] - \
                                multi_sub_landmarks["bbox"][frame_idx][temp_person_id][0] - \
                                multi_sub_landmarks["bbox"][frame_idx][temp_person_id][1]
                if temp_bbox_len > max_bbox_len:
                    max_bbox_person_id = temp_person_id
                    max_bbox_len = temp_bbox_len
                    landmark_scores = multi_sub_landmarks['landmarks_scores'][frame_idx][temp_person_id]
            if landmark_scores[17:].min() >= 0.2:
                landmarks[frame_idx] = multi_sub_landmarks["landmarks"][frame_idx][max_bbox_person_id]
    return landmarks
