#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain
from bob.gradiant.pipelines.classes.processor import Processor
from sklearn.feature_selection import RFECV, RFE
from sklearn.svm import LinearSVC
import pickle
import numpy as np
import copy


class RecursiveFeatureElimination(Processor):
    def __init__(self, name='rfe', C=1.0, n_features=10, step=1, verbose=0):
        super(RecursiveFeatureElimination, self).__init__(name)
        self.n_features = n_features
        self.model_loaded = False
        linear_svc = LinearSVC(C=C)
        if n_features == 0:
            self._model = RFECV(linear_svc, cv=3, step=step, verbose=verbose)
        else:
            self._model = RFE(linear_svc, n_features_to_select=n_features, step=step, verbose=verbose)

    def to_dict(self):
        output_dict = {
            'data': np.array(pickle.dumps(self._model)),
        }

        return output_dict

    def from_dict(self, dict):
        self._model = pickle.loads(dict['data'])

    def fit(self, X):
        if self.model_loaded is True:
            return
        labels = copy.deepcopy(X['labels'])
        labels[labels > 0] = 1
        self._model.fit(X['features'], labels)

    def run(self, X):
        X['features'] = self._model.transform(X['features'])
        X['indices'] = np.where(self._model.ranking_ == 1)[0]
        return X

    def __str__(self):
        if self.model_loaded:
            description = {'type': 'Feature selection processor',
                           'loaded': 'from_file'}
        else:
            description = {
                'type': 'Feature selection processor',
                'n_features': self.n_features,
                'name': self.name
            }
        return str(description)

    def load(self, base_path):
        super(RecursiveFeatureElimination, self).load(base_path)
        self.model_loaded = True
