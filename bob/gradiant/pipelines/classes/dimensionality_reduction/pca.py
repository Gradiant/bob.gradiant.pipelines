#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

from bob.gradiant.pipelines.classes.processor import Processor
from sklearn.decomposition.pca import PCA
import pickle
import numpy as np


class Pca(Processor):
    _model = None

    def __init__(self, name='pca', n_components=None):
        super(Pca, self).__init__(name)
        self._model = PCA(n_components=n_components)

    def fit(self, X):
        self._model.fit(X['features'])

    def run(self, X):
        X['features'] = self._model.transform(X['features'])
        return X

    def from_dict(self, dict):
        self._model = pickle.loads(dict['data'])

    def to_dict(self):
        dict = {
            'data': np.array(pickle.dumps(self._model))
        }
        return dict

    def __str__(self):
        description = {
            'type': 'PCA',
            'name': self.name,
            'n_components': self._model.n_components
        }
        return str(description)
