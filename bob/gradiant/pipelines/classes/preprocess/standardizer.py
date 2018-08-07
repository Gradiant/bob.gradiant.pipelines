#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

from bob.gradiant.pipelines.classes.processor import Processor
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np


class Standardizer(Processor):
    _model = None

    def __init__(self, name='standardizer'):
        super(Standardizer, self).__init__(name)
        self._model = StandardScaler()

    def fit(self, X):
        self._model.fit(X['features'])

    def run(self, X):
        X['features'] = self._model.transform(X['features'])
        return X

    def to_dict(self):
        output_dict = {
            'data': np.array(pickle.dumps(self._model)),
        }

        return output_dict

    def from_dict(self, dict):
        self._model = pickle.loads(dict['data'])

    def printmodel(self):
        print(self._model)

    def to_dict(self):
        dict = {
            'data': np.array(pickle.dumps(self._model))
        }
        return dict

    def __str__(self):
        description = {
            'type': 'Standard Scaler',
            'name': self.name
        }
        return str(description)
