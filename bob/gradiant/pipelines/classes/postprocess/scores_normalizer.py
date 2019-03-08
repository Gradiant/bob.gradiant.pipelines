#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.default_keys_correspondences import DEFAULT_KEYS_CORRESPONDENCES

import pickle
import numpy as np


def normalize(scores, min_value=None, max_value=None):
    if min_value is None:
        min_value = min(scores)
    if max_value is None:
        max_value = max(scores)

    list_normalized_scores = []
    for value in scores.tolist():
        normalized_score = (value - min_value) / (max_value - min_value)
        if normalized_score >= 1:
            normalized_score = 1.0
        if normalized_score < 0:
            normalized_score = 0
        list_normalized_scores.append(normalized_score)
    return np.array(list_normalized_scores)


class ScoresNormalizer(Processor):
    def __init__(self,
                 name='scores_normalizer',
                 keys_correspondences=DEFAULT_KEYS_CORRESPONDENCES):
        super(ScoresNormalizer, self).__init__(name)
        self.keys_correspondences = keys_correspondences

    def fit(self, x):
        pass

    def run(self, x):
        scores_key = self.keys_correspondences["scores_key"]

        normalized_scores = normalize(x[scores_key])
        x[scores_key] = normalized_scores
        return x

    def to_dict(self):
        return None

    def from_dict(self, dict):
        pass

    def printmodel(self):
        print(self._model)

    def save(self, base_path):
        pass

    def load(self, base_path):
        pass

    def __str__(self):
        description = {
            'name': self.name
        }
        return str(description)
