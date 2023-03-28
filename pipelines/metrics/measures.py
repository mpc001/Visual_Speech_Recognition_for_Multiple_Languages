#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This code refers https://github.com/espnet/espnet/blob/24c3676a8d4c2e60d2726e9bcd9bdbed740610e0/espnet/nets/e2e_asr_common.py#L213-L249

import numpy as np

def get_wer(s, ref):
    return get_er(s.split(), ref.split())

def get_cer(s, ref):
    return get_er(s.replace(" ", ""), ref.replace(" ", ""))

def get_er(s, ref):
    """
        FROM wikipedia levenshtein distance
        s: list of words/char in sentence to measure
        ref: list of words/char in reference
    """

    costs = np.zeros((len(s) + 1, len(ref) + 1))
    for i in range(len(s) + 1):
        costs[i, 0] = i
    for j in range(len(ref) + 1):
        costs[0, j] = j

    for j in range(1, len(ref) + 1):
        for i in range(1, len(s) + 1):
            cost = None
            if s[i-1] == ref[j-1]:
                cost = 0
            else:
                cost = 1
            costs[i,j] = min(
                costs[i-1, j] + 1,
                costs[i, j-1] + 1,
                costs[i-1, j-1] + cost
            )

    return costs[-1,-1] / len(ref)
