#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import torch
import hydra
from pipelines.metrics.measures import get_wer
from pipelines.metrics.measures import get_cer
from pipelines.pipeline import InferencePipeline


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.total = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.total += val * n
        self.count += n
        self.avg = self.total / self.count


def benchmark_inference(inference_pipeline, data_dir, landmarks_dir, lines, data_ext=".mp4", landmarks_ext=".pkl"):
    wer, cer = AverageMeter(), AverageMeter()
    for idx, line in enumerate(lines):
        basename, groundtruth = line.split()[0], " ".join(line.split()[1:])
        data_filename = os.path.join(data_dir, f"{basename}{data_ext}")
        landmarks_filename = os.path.join(landmarks_dir, f"{basename}{landmarks_ext}") if landmarks_dir else None
        output = inference_pipeline(data_filename, landmarks_filename)

        print(f"hyp: {output}\nref: {groundtruth}" if groundtruth is not None else "")
        if groundtruth is not None:
            wer.update(get_wer(output, groundtruth), len(groundtruth.split()))
            cer.update(get_cer(output, groundtruth), len(groundtruth))
            print(f"progress: {idx+1}/{len(lines)}\tcur WER: {wer.val*100:.2f}\tcur CER: {cer.val*100:.2f}\tavg WER: {wer.avg*100:.2f}\tavg CER: {cer.avg*100:.2f}")


@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    device = torch.device(f"cuda:{cfg.gpu_idx}") if torch.cuda.is_available() and cfg.gpu_idx >= 0 else "cpu"
    inference_pipeline = InferencePipeline(config_filename=cfg.config_filename, detector=cfg.detector, face_track=not cfg.landmarks_filename and not cfg.landmarks_dir, device=device)
    assert os.path.isdir(cfg.data_dir), f"{cfg.data_dir} is not a directory."
    assert os.path.isfile(cfg.labels_filename), f"{cfg.labels_filename} does not exist."
    benchmark_inference(inference_pipeline, cfg.data_dir, cfg.landmarks_dir, open(cfg.labels_filename).read().splitlines(), cfg.data_ext, cfg.landmarks_ext)


if __name__ == '__main__':
    main()
