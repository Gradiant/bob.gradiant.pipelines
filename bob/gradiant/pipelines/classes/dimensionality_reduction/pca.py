#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.default_keys_correspondences import DEFAULT_KEYS_CORRESPONDENCES
from sklearn.decomposition.pca import PCA
import sys
import pickle
import numpy as np


class Pca(Processor):
    _model = None

    def __init__(self,
                 name='pca',
                 n_components=None,
                 keys_correspondences=DEFAULT_KEYS_CORRESPONDENCES):
        super(Pca, self).__init__(name)
        self._model = PCA(n_components=n_components)
        self.keys_correspondences = keys_correspondences

    def fit(self, x):
        self._model.fit(x['features'])

    def run(self, x):
        features_key = self.keys_correspondences["features_key"]

        x[features_key] = self._model.transform(x[features_key])
        return x

    def from_dict(self, dict):
        if sys.version_info[0] < 3:
            self._model = pickle.loads(dict['data'])
        else:
            self._model = pickle.loads(dict['data'], encoding='latin1')

    def to_dict(self):
        dict = {
            'data': np.array(pickle.dumps(self._model))
        }
        return dict

    def __str__(self):
        description = {
            'type': 'Dimensionality reduction',
            'name': self.name,
            'n_components': self._model.n_components
        }
        return str(description)
