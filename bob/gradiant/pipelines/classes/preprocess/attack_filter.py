#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.default_keys_correspondences import DEFAULT_KEYS_CORRESPONDENCES
from bob.gradiant.pipelines.classes.processor_output_type import ProcessorOutputType
import numpy as np


class AttackFilter(Processor):
    _index_filter = []

    def __init__(self,
                 name='attack_filter',
                 index_filter=None,
                 keys_correspondences=DEFAULT_KEYS_CORRESPONDENCES):
        super(AttackFilter, self).__init__(name)
        if index_filter is None:
            self._index_filter = [1]
        else:
            self._index_filter = index_filter
        self.keys_correspondences = keys_correspondences

    def fit(self, x):
        pass

    def run(self, x):
        labels_key = self.keys_correspondences["labels_key"]
        features_key = self.keys_correspondences["features_key"]
        access_ids_key = self.keys_correspondences["access_ids_key"]
        output_type_key = self.keys_correspondences["output_type_key"]

        labels = x[labels_key]
        ind_attack = np.array([])
        for i in self._index_filter:
            ind_attack = np.concatenate((ind_attack, np.where(labels == i)[0]), axis=0)

        ind_gen = np.where(labels == 0)[0]
        indices = np.uint(np.concatenate((ind_gen, ind_attack), axis=0))

        # for k, v in X.items(): X[k] = X[k][indices]
        x[features_key] = x[features_key][indices]
        x[access_ids_key] = x[access_ids_key][indices]
        x[labels_key] = x[labels_key][indices]
        x[output_type_key] = ProcessorOutputType.LIKELIHOOD
        return x

    @staticmethod
    def to_dict():
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
