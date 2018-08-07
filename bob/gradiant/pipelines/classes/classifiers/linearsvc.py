#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.processor_output_type import ProcessorOutputType
from sklearn.svm import LinearSVC
import pickle
import numpy as np


class LinearSvc(Processor):
    """
      This class encapsulates sklearn.svm.LinearSVC.

      **Parameters:**

      ``name`` : :py:class:`str`
          Name for LinearSVC. Default: 'linear_svc'.

      ``C`` : :py:class:`float`
          Cost function. Default: 1.0.
      """

    _model = None

    def __init__(self, name='linear_svc', C=1.0):
        super(LinearSvc, self).__init__(name)
        self._model = LinearSVC(C=C)

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
            'type': 'Linear SVM',
            'name': self.name,
            'C': self._model.C
        }
        return str(description)
