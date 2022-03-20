#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import os
import time
from configparser import ConfigParser

from lipreading.model import Lipreading
from dataloader.dataloader import AVSRDataLoader
from dataloader.utils import get_video_properties


class LipreadingPipeline(object):
    """LipreadingPipeline."""

    def __init__(self,
        config_filename,
        feats_position=None,
        face_track=False,
        device="cpu",
    ):
        """__init__.

        :param config_filename: str, the filename of the configuration.
        :param feats_position: str, the layer position for feature extraction.
        :param face_track: str, face tracker will be used if set it as True.
        :param device: str, contain the device on which a torch.Tensor is or will be allocated.
        """
        if feats_position == "mouth":
            self.modality = "video"
            self.dataloader = AVSRDataLoader(
                modality=self.modality,
                disable_transform=True,
            )
            self.model = None
        else:
            assert os.path.isfile(config_filename), \
                f"config_filename: {config_filename} does not exist."
            config = ConfigParser()
            config.read(config_filename)
            self.input_v_fps = config.getfloat("input", "v_fps")
            self.model_v_fps = config.getfloat("model", "v_fps")
            self.modality = config.get("input", "modality")

            self.dataloader = AVSRDataLoader(
                modality=self.modality,
                speed_rate=self.input_v_fps/self.model_v_fps,
            )
            self.model = Lipreading(
                config,
                feats_position=feats_position,
                device=device,
            )

        if face_track and self.modality == "video":
            from tracker.face_tracker import FaceTracker
            self.face_tracker = FaceTracker(device="cuda:0")
        else:
            self.face_tracker = None

        self.feats_position = feats_position


    def __call__(
        self,
        data_filename,
        landmarks_filename,
    ):
        """__call__.

        :param data_filename: str, the filename of the input sequence.
        :param landmarks_filename: str, the filename of the corresponding landmarks.
        """
        # Step 1, track face in the input video or read landmarks from the file.
        assert os.path.isfile(data_filename), \
            f"data_filename: {data_filename} does not exist."

        if self.modality == "audio":
            landmarks = None
        else:
            if os.path.isfile(landmarks_filename):
                landmarks = landmarks_filename
            else:
                assert self.face_tracker is not None, "face tracker is not enabled."
                end=time.time()
                landmarks = self.face_tracker.tracker(data_filename)
                print(f"face tracking speed: {len(landmarks)/(time.time()-end):.2f} fps.")

        # Step 2, extract mouth patches from segments.
        sequence = self.dataloader.load_data(
            data_filename,
            landmarks,
        )

        # Step 3, perform inference or extract mouth ROIs or speech representations.
        if self.feats_position:
            if self.feats_position == "mouth":
                assert self.modality == "video", "input modality should be `video`."
                vid_fps = get_video_properties(data_filename)["fps"]
                output = (sequence, vid_fps)
            else:
                output = self.model.extract_feats(sequence)
        else:
            output = self.model.predict(sequence)
        return output
