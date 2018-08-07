#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.processor_output_type import ProcessorOutputType
from sklearn.svm import SVC
import pickle
import numpy as np


class RbfSvc(Processor):
    _model = None

    def __init__(self, name='rbf_svc', C=1.0, gamma='auto'):
        super(RbfSvc, self).__init__(name)
        self._model = SVC(C=C, gamma=gamma)

    def fit(self, X):
        labels = np.clip(X['labels'], 0, 1)
        labels = -labels + 1
        self._model.fit(X['features'], labels)

    def run(self, X):
        X['scores'] = self._model.decision_function(X['features'])
        X['output_type'] = ProcessorOutputType.LIKELIHOOD
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
            'type': 'RBF SVM',
            'name': self.name,
            'C': self._model.C,
            'gamma': self._model.gamma
        }
        return str(description)
