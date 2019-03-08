#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain
from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.default_keys_correspondences import DEFAULT_KEYS_CORRESPONDENCES
from sklearn.feature_selection import RFECV, RFE
from sklearn.svm import LinearSVC
import pickle
import numpy as np
import copy


class RecursiveFeatureElimination(Processor):
    def __init__(self,
                 name='rfe',
                 c=1.0,
                 n_features=10,
                 step=1,
                 verbose=0,
                 keys_correspondences=DEFAULT_KEYS_CORRESPONDENCES):
        super(RecursiveFeatureElimination, self).__init__(name)
        self.n_features = n_features
        self.model_loaded = False
        linear_svc = LinearSVC(C=c)
        if n_features == 0:
            self._model = RFECV(linear_svc, cv=3, step=step, verbose=verbose)
        else:
            self._model = RFE(linear_svc, n_features_to_select=n_features, step=step, verbose=verbose)
        self.keys_correspondences = keys_correspondences

    def to_dict(self):
        output_dict = {
            'data': np.array(pickle.dumps(self._model)),
        }
        return output_dict

    def from_dict(self, dict):
        self._model = pickle.loads(dict['data'])

    def fit(self, x):
        if self.model_loaded is True:
            return

        labels_key = self.keys_correspondences["labels_key"]
        features_key = self.keys_correspondences["features_key"]

        labels = copy.deepcopy(x[labels_key])
        labels[labels > 0] = 1
        self._model.fit(x[features_key], labels)

    def run(self, x):
        features_key = self.keys_correspondences["features_key"]
        indices_key = self.keys_correspondences["indices_key"]

        x[features_key] = self._model.transform(x[features_key])
        x[indices_key] = np.where(self._model.ranking_ == 1)[0]
        return x

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
