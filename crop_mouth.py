#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import cv2
import hydra
import torchvision
from pipelines.detectors.mediapipe.detector import LandmarksDetector
from pipelines.data.data_module import AVSRDataLoader


def save2vid(filename, vid, frames_per_second):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchvision.io.write_video(filename, vid, frames_per_second)


@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    if cfg.detector == "mediapipe":
        from pipelines.detectors.mediapipe.detector import LandmarksDetector
        landmarks_detector = LandmarksDetector()
    if cfg.detector == "retinaface":
        from pipelines.detectors.retinaface.detector import LandmarksDetector
        landmarks_detector = LandmarksDetector()
    dataloader = AVSRDataLoader(modality="video", speed_rate=1, transform=False, detector=cfg.detector, convert_gray=False)
    landmarks = landmarks_detector(cfg.data_filename)
    data = dataloader.load_data(cfg.data_filename, landmarks)
    fps = cv2.VideoCapture(cfg.data_filename).get(cv2.CAP_PROP_FPS)
    save2vid(cfg.dst_filename, data, fps)
    print(f"The mouth images have been cropped and saved to {cfg.dst_filename}")


if __name__ == "__main__":
    main()
