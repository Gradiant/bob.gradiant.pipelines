#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.default_keys_correspondences import DEFAULT_KEYS_CORRESPONDENCES

import numpy as np


class ScoresZNormalizer(Processor):
    def __init__(self,
                 name='scores_z_normalizer',
                 n_sigma=3,
                 keys_correspondences=DEFAULT_KEYS_CORRESPONDENCES):
        super(ScoresZNormalizer, self).__init__(name)
        self.keys_correspondences = keys_correspondences
        self.n_sigma = n_sigma
        self.mean = 0.0
        self.std = 0.0

    def fit(self, x):
        scores_key = self.keys_correspondences["scores_key"]
        scores = x[scores_key].ravel()
        self.mean = np.mean(scores)
        self.std = np.std(scores)

    def run(self, x):
        scores_key = self.keys_correspondences["scores_key"]

        # Apply Z-normalization
        scores = x[scores_key].ravel()
        scores -= self.mean
        scores /= self.std

        # Truncate to (-n-sigma, n-sigma)
        scores = np.clip(scores, -1*self.n_sigma, self.n_sigma)

        # Move to interval 0-1 (genuine probability)
        scores += self.n_sigma
        scores /= 2*self.n_sigma

        x[scores_key] = scores
        return x

    def to_dict(self):
        dict = {
            'mean': np.array([self.mean]),
            'std': np.array([self.std]),
            'n_sigma': np.array([self.n_sigma])
        }
        return dict

    def from_dict(self, dict):
        self.mean = dict['mean'].item(),
        self.std = dict['std'].item(),
        self.n_sigma = dict['n_sigma'].item()

    def __str__(self):
        description = {
            'type': 'Postprocessor',
            'name': self.name,
            'mean (Train)': self.mean,
            'std (Train)': self.std,
            'n_sigma': self.n_sigma
        }
        return str(description)
