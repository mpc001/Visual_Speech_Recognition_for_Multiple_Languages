#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import cv2
import pickle
import numpy as np
from .preprocess import *
from .transform import *
from .utils import *


class AVSRDataLoader(object):
    """AVSRDataLoader.
    Customised DataLoader to get raw audio waveforms or mouth RoIs.

    """ 

    def __init__(
        self,
        modality,
        disable_transform=False,
        speed_rate=1.0,
        mean_face_path="20words_mean_face.npy",
        noise_file_path="babbleNoise_resample_16K.npy",
        crop_width=96,
        crop_height=96,
        start_idx=48,
        stop_idx=68,
        window_margin=12,
        convert_gray=True,
    ):
        """__init__.

        :param modality: str, the modality type of the input.
        :param disable_transform: bool, disable transform if set as True.
        :param speed_rate: float, the speed rate between the frame rate of \
            the input video and the frame rate used for training.
        :param mean_face_path: str, the file path of the reference face.
        :param noise_file_path: str, the file path of the noisy background.
        :param crop_width: int, the width of the cropped patch.
        :param crop_height: int, the height of the cropped patch.
        :param start_idx: int, the starting index for cropping the bounding box.
        :param stop_idx: int, the ending inex for cropping the bounding box.
        :param window_margin: int, the window size for smoothing landmarks.
        :param convert_gray: bool, save as grayscale if set as True.
        """
        self.modality = modality
        self.disable_transform = disable_transform
        self._reference = np.load(
            os.path.join( os.path.dirname(__file__), mean_face_path)
        )
        self._noise = np.load(
            os.path.join( os.path.dirname(__file__), noise_file_path)
        )
        self._crop_width = crop_width
        self._crop_height = crop_height
        self._start_idx = start_idx
        self._stop_idx = stop_idx
        self._window_margin = window_margin
        self._convert_gray = convert_gray

        if self.modality == "audio":
            self.transform = self.get_audio_transform(self._noise)
        elif self.modality == "video":
            self.transform = self.get_video_transform(speed_rate=speed_rate)


    def get_audio_transform(self, noise_data):
        """get_audio_transform.

        :param noise_data: numpy.ndarray, the noisy data to be injected to data.
        """

        return Compose([
            AddNoise(
                noise=noise_data,
                snr_target=9999,
            ),
            NormalizeUtterance(),
            ExpandDims()]
        )


    def get_video_transform(self, speed_rate=1.0):
        """get_video_transform.

        :param speed_rate: float, the speed rate between the frame rate of \
            the input video and the frame rate used for training.
        """
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
       	return Compose([
            Normalize(0.0, 255.0),
            CenterCrop(crop_size),
            Normalize(mean, std),
            SpeedRate(
                speed_rate=speed_rate,
            ) if speed_rate != 1.0 else Identity()]
        )


    def preprocess(self, video_pathname, landmarks_pathname):
        """preprocess.

        :param video_pathname: str, the filename for the video.
        :param landmarks_pathname: str, the filename for the landmarks.
        """
        # -- Step 1, extract landmarks from pkl files.
        if isinstance(landmarks_pathname, str):
            with open(landmarks_pathname, "rb") as pkl_file:
                landmarks = pickle.load(pkl_file)
        else:
            landmarks = landmarks_pathname
        # -- Step 2, pre-process landmarks: interpolate frames that not being detected.
        preprocessed_landmarks = self.landmarks_interpolate(landmarks)
        # -- Step 3, exclude corner cases:
        #   -- 1) no landmark in all frames
        #   -- 2) number of frames is less than window length.
        if not preprocessed_landmarks or len(preprocessed_landmarks) < self._window_margin: return
        # -- Step 4, affine transformation and crop patch 
        sequence, transformed_frame, transformed_landmarks = \
            self.crop_patch(video_pathname, preprocessed_landmarks)
        assert sequence is not None, "cannot crop from {}.".format(filename)
        return sequence


    def landmarks_interpolate(self, landmarks):
        """landmarks_interpolate.

        :param landmarks: List, the raw landmark (in-place)

        """
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        if not valid_frames_idx:
            return None
        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
                continue
            else:
                landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        # -- Corner case: keep frames at the beginning or at the end failed to be detected.
        if valid_frames_idx:
            landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
        return landmarks


    def crop_patch(self, video_pathname, landmarks):
        """crop_patch.

        :param video_pathname: str, the filename for the processed video.
        :param landmarks: List, the interpolated landmarks.
        """

        frame_idx = 0
        frame_gen = load_video(video_pathname)
        while True:
            try:
                frame = frame_gen.__next__() ## -- BGR
            except StopIteration:
                break
            if frame_idx == 0:
                sequence = []
                sequence_frame = []
                sequence_landmarks = []

            window_margin = min(self._window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed_landmarks = np.mean([landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)], axis=0)
            smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
            transformed_frame, transformed_landmarks = self.affine_transform(
                frame,
                smoothed_landmarks,
                self._reference,
                grayscale=self._convert_gray,
            )
            sequence.append( cut_patch( transformed_frame,
                                        transformed_landmarks[self._start_idx:self._stop_idx],
                                        self._crop_height//2,
                                        self._crop_width//2,))

            sequence_frame.append( transformed_frame)
            sequence_landmarks.append( transformed_landmarks)
            frame_idx += 1

        return np.array(sequence), np.array(sequence_frame), np.array(sequence_landmarks)


    def affine_transform(
        self,
        frame,
        landmarks,
        reference,
        grayscale=False,
        target_size=(256, 256),
        reference_size=(256, 256),
        stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0
    ):
        """affine_transform.

        :param frame: numpy.array, the input sequence.
        :param landmarks: List, the tracked landmarks.
        :param reference: numpy.array, the neutral reference frame.
        :param grayscale: bool, save as grayscale if set as True.
        :param target_size: tuple, size of the output image.
        :param reference_size: tuple, size of the neural reference frame.
        :param stable_points: tuple, landmark idx for the stable points.
        :param interpolation: interpolation method to be used.
        :param border_mode: Pixel extrapolation method .
        :param border_value: Value used in case of a constant border. By default, it is 0.
        """
        # Prepare everything
        if grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stable_reference = np.vstack([reference[x] for x in stable_points])
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0

        # Warp the face patch and the landmarks
        transform = cv2.estimateAffinePartial2D(np.vstack([landmarks[x] for x in stable_points]),
                                                stable_reference, method=cv2.LMEDS)[0]
        transformed_frame = cv2.warpAffine(
            frame,
            transform,
            dsize=(target_size[0], target_size[1]),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
        transformed_landmarks = np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()

        return transformed_frame, transformed_landmarks


    def load_audio(self, data_filename):
        """load_audio.

        :param data_filename: str, the filename of input sequence.
        """
        sequence = load_audio(data_filename, specified_sr=16000, int_16=False)
        return sequence


    def load_video(self, data_filename, landmarks_filename=None):
        """load_video.

        :param data_filename: str, the filename of input sequence.
        :param landmarks_filename: str, the filename of landmarks.
        """
        assert landmarks_filename is not None
        sequence = self.preprocess(
            video_pathname=data_filename,
            landmarks_pathname=landmarks_filename,
        )
        return sequence


    def load_data(self, data_filename, landmarks_filename=None):
        """load_data.

        :param data_filename: str, the filename of input sequence.
        :param landmarks_filename: str, the filename of landmarks.
        """
        if self.modality == "audio":
            raw_sequence = self.load_audio(data_filename)
        elif self.modality == "video":
            raw_sequence = self.load_video(
                data_filename,
                landmarks_filename=landmarks_filename,
            )
        if self.disable_transform:
            return raw_sequence
        else:
            return self.transform(raw_sequence)
