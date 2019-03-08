#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.default_keys_correspondences import DEFAULT_KEYS_CORRESPONDENCES
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np


class Standardizer(Processor):
    _model = None

    def __init__(self,
                 name='standardized',
                 keys_correspondences=DEFAULT_KEYS_CORRESPONDENCES):
        super(Standardizer, self).__init__(name)
        self._model = StandardScaler()
        self.keys_correspondences = keys_correspondences

    def fit(self, x):
        features_key = self.keys_correspondences["features_key"]

        self._model.fit(x[features_key])

    def run(self, x):
        features_key = self.keys_correspondences["features_key"]

        x[features_key] = self._model.transform(x[features_key])
        return x

    def to_dict(self):
        output_dict = {
            'data': np.array(pickle.dumps(self._model)),
        }

        return output_dict

    def from_dict(self, dict):
        self._model = pickle.loads(dict['data'])

    def printmodel(self):
        print(self._model)

    def __str__(self):
        description = {
            'type': 'Standard Scaler',
            'name': self.name
        }
        return str(description)
