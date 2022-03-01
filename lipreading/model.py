#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import os
import json
import torch
import argparse
import numpy as np

from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import add_results_to_json
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E


class Lipreading(torch.nn.Module):
    """Lipreading."""

    def __init__(self, config, feats_position="resnet", device="cpu"):
        """__init__.

        :param config: ConfigParser class, contains model's configuration.
        :param feats_position: str, the position to extract features.
        :param device: str, contain the device on which a torch.Tensor is or will be allocated.
        """
        super(Lipreading, self).__init__()

        self.device = device
        self.feats_position = feats_position

        self.load_model(config)
        self.get_beam_search(config)

        self.model.to(device=self.device).eval()
        self.beam_search.to(device=self.device).eval()


    def load_model(self, config):
        """load_model.

        :param config: ConfigParser class, the configuration parser.
        """
        model_path = config.get("model","model_path")
        model_conf = config.get("model","model_conf")

        assert os.path.isfile(model_path), f"model_path: {model_path} does not exist."
        assert os.path.isfile(model_conf), f"model_conf: {model_conf} does not exist."

        with open(model_conf, "rb") as f:
            confs = json.load(f)
        if isinstance(confs, dict):
            args = confs
        else:
            idim, odim, args = confs
            self.odim = odim
        self.train_args = argparse.Namespace(**args)
        self.char_list = self.train_args.char_list
        self.model = E2E(odim, self.train_args)
        
        # -- load a pre-trained model
        self.model.load_state_dict(torch.load(model_path))
        print(f"load a pre-trained model from: {model_path}")

        
    def get_beam_search(self, config):
        """get_beam_search.

        :param config: ConfigParser Objects, the main configuration parser.
        """

        rnnlm = config.get("model","rnnlm")
        rnnlm_conf = config.get("model","rnnlm_conf")

        penalty = config.getfloat("decode", "penalty")
        maxlenratio = config.getfloat("decode", "maxlenratio")
        minlenratio = config.getfloat("decode", "minlenratio")
        ctc_weight = config.getfloat("decode", "ctc_weight")
        lm_weight = config.getfloat("decode", "lm_weight")
        beam_size = config.getint("decode", "beam_size")

        sos = self.odim - 1
        eos = self.odim - 1
        scorers = self.model.scorers()

        if not rnnlm:
            lm = None
        else:
            lm_args = get_model_conf(rnnlm, rnnlm_conf)
            lm_model_module = getattr(lm_args, "model_module", "default")
            lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
            lm = lm_class(len(self.train_args.char_list), lm_args)
            torch_load(rnnlm, lm)
            lm.eval()

        scorers["lm"] = lm
        scorers["length_bonus"] = LengthBonus(len(self.train_args.char_list))
        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            length_bonus=penalty,
        )

        # -- decoding config
        self.beam_size = beam_size
        self.nbest = 1
        self.weights = weights
        self.scorers = scorers
        self.sos = sos
        self.eos = eos
        self.ctc_weight = ctc_weight
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio

        self.beam_search = BatchBeamSearch(
            beam_size=self.beam_size,
            vocab_size=len(self.train_args.char_list),
            weights=self.weights,
            scorers=self.scorers,
            sos=self.sos,
            eos=self.eos,
            token_list=self.train_args.char_list,
            pre_beam_score_key=None if self.ctc_weight == 1.0 else "decoder",
        )


    def predict(self, sequence):
        """predict.

        :param sequence: ndarray, the raw sequence saved in a format of numpy array.
        """
        with torch.no_grad():
            sequence = (torch.FloatTensor(sequence).to(self.device))
            enc_feats = self.model.encode(torch.as_tensor(sequence).to(device=self.device))
            nbest_hyps = self.beam_search(
                x=enc_feats,
                maxlenratio=self.maxlenratio,
                minlenratio=self.minlenratio
                )
            nbest_hyps = [
                h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), self.nbest)]
            ]

            transcription = add_results_to_json(nbest_hyps, self.char_list)

        return transcription.replace("<eos>", "")


    def extract_feats(self, sequence):
        """extract_feats.

        :param sequence: ndarray, the raw sequence saved in a format of numpy array.
        """
        sequence = (torch.FloatTensor(sequence).to(self.device))
        if self.feats_position == "resnet":
            feats = self.model.encode(
                torch.as_tensor(sequence).to(device=self.device),
                extract_resnet_feats=True,
            )
        elif self.feats_position == "conformer":
            feats = self.model.encode(torch.as_tensor(sequence).to(device=self.device))
        return feats
