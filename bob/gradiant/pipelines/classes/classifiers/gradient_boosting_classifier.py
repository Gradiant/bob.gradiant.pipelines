#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain
from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.processor_output_type import ProcessorOutputType
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
import pickle
import numpy as np
import copy


class GradientBoosting(Processor):
    def __init__(self, name='rfe', C=1.0, n_estimators=100):
        super(GradientBoosting, self).__init__(name)
        self._model = GradientBoostingClassifier(LinearSVC(C=C), n_estimators=n_estimators, max_depth=1000)

    def to_dict(self):
        output_dict = {
            'data': np.array(pickle.dumps(self._model)),
        }

        return output_dict

    def from_dict(self, dict):
        self._model = pickle.loads(dict['data'])

    def fit(self, X):
        labels = copy.deepcopy(X['labels'])
        labels[labels > 0] = 1
        self._model.fit(X['features'], labels)

    def run(self, X):
        X['scores'] = self._model.decision_function(X['features'])
        X['output_type'] = ProcessorOutputType.LIKELIHOOD
        return X

    def __str__(self):
        description = {
            'type': 'Gradient boosting processor',
            'name': self.name
        }
        return str(description)
