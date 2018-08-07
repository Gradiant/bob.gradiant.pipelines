#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.processor_output_type import ProcessorOutputType
import pickle
import numpy as np


class AttackFilter(Processor):
    _index_filter = []

    def __init__(self, name='attack_filter', index_filter=None):
        super(AttackFilter, self).__init__(name)
        if index_filter is None:
            self._index_filter = [1]
        else:
            self._index_filter = index_filter

    def fit(self, X):
        pass

    def run(self, X):
        labels = X['labels']
        ind_attack = np.array([])
        for i in self._index_filter:
            ind_attack = np.concatenate((ind_attack, np.where(labels == i)[0]), axis=0)

        ind_gen = np.where(labels == 0)[0]
        indices = np.uint(np.concatenate((ind_gen, ind_attack), axis=0))

        # for k, v in X.iteritems(): X[k] = X[k][indices]
        X['features'] = X['features'][indices]
        X['access_ids'] = X['access_ids'][indices]
        X['labels'] = X['labels'][indices]
        X['output_type'] = ProcessorOutputType.LIKELIHOOD
        return X

    def to_dict(self):
        return None

    def from_dict(self, dict):
        pass

    def save(self, base_path):
        pass

    def load(self, base_path):
        pass

    def __str__(self):
        description = {
            'type': 'attack type filter',
            'name': self.name
        }
        return str(description)
