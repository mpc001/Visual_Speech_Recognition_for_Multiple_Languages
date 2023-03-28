#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
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
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E


class AVSR(torch.nn.Module):
    def __init__(self, config, device="cpu"):
        super(AVSR, self).__init__()

        self.device = device

        self.load_model(config)
        self.model.to(device=self.device).eval()

        beam_search_decoder = BeamSearchDecoder(self.model, self.token_list, config)
        self.beam_search = beam_search_decoder.get_batch_beam_search()
        self.beam_search.to(device=self.device).eval()


    def load_model(self, config):
        if config.get("input", "modality") == "audiovisual":
            from espnet.nets.pytorch_backend.e2e_asr_transformer_av import E2E
        else:
            from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E

        model_path = config.get("model","model_path")
        model_conf = config.get("model","model_conf")

        assert os.path.isfile(model_path), f"model_path: {model_path} does not exist."
        assert os.path.isfile(model_conf), f"model_conf: {model_conf} does not exist."

        with open(model_conf, "rb") as f:
            confs = json.load(f)
        args = confs if isinstance(confs, dict) else confs[2]
        self.train_args = argparse.Namespace(**args)

        labels_type = getattr(self.train_args, "labels_type", "char")
        if labels_type == "char":
            self.token_list = self.train_args.char_list
        elif labels_type == "unigram5000":
            file_path = os.path.join(os.path.dirname(__file__), "tokens", "unigram5000_units.txt")
            self.token_list = ['<blank>'] + [word.split()[0] for word in open(file_path).read().splitlines()] + ['<eos>']
        self.odim = len(self.token_list)

        # Initialize and load the pre-trained model
        self.model = E2E(self.odim, self.train_args)
        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

        
    def infer(self, data):
        with torch.no_grad():
            if isinstance(data, tuple):
                enc_feats = self.model.encode(data[0].to(self.device), data[1].to(self.device))
            else:
                enc_feats = self.model.encode(data.to(self.device))
            nbest_hyps = self.beam_search(enc_feats)
            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
            transcription = add_results_to_json(nbest_hyps, self.token_list)
            transcription = transcription.replace("▁", " ").strip()
        return transcription.replace("<eos>", "")


class BeamSearchDecoder:
    def __init__(self, model, token_list, config):
        self.model = model
        self.odim = model.odim
        self.token_list = token_list
        self.config = config

    def get_batch_beam_search(self):
        rnnlm = self.config.get("model", "rnnlm")
        rnnlm_conf = self.config.get("model", "rnnlm_conf")

        penalty = self.config.getfloat("decode", "penalty")
        ctc_weight = self.config.getfloat("decode", "ctc_weight")
        lm_weight = self.config.getfloat("decode", "lm_weight")
        beam_size = self.config.getint("decode", "beam_size")

        sos = self.odim - 1
        eos = self.odim - 1
        scorers = self.model.scorers()

        if not rnnlm:
            lm = None
        else:
            lm_args = get_model_conf(rnnlm, rnnlm_conf)
            lm_model_module = getattr(lm_args, "model_module", "default")
            lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
            lm = lm_class(len(self.token_list), lm_args)
            torch_load(rnnlm, lm)
            lm.eval()

        scorers["lm"] = lm
        scorers["length_bonus"] = LengthBonus(len(self.token_list))
        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            length_bonus=penalty,
        )

        return BatchBeamSearch(
            beam_size=beam_size,
            vocab_size=len(self.token_list),
            weights=weights,
            scorers=scorers,
            sos=sos,
            eos=eos,
            token_list=self.token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
        )
