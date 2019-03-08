#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain
import sys
import pickle
import numpy as np

from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.default_keys_correspondences import DEFAULT_KEYS_CORRESPONDENCES
from bob.gradiant.pipelines.classes.processor_output_type import ProcessorOutputType
from sklearn.svm import LinearSVC


class LinearSvc(Processor):
    """
      This class encapsulates sklearn.svm.LinearSVC.

      **Parameters:**

      ``name`` : :py:class:`str`
          Name for LinearSVC. Default: 'linear_svc'.

      ``C`` : :py:class:`float`
          Cost function. Default: 1.0.

      ``keys_correspondences`` : :py:class:`dict`
          key for access to values on X. Default: {}.

      """.format(DEFAULT_KEYS_CORRESPONDENCES)

    _model = None

    def __init__(self, name='linear_svc',
                 c=1.0,
                 keys_correspondences=DEFAULT_KEYS_CORRESPONDENCES):
        super(LinearSvc, self).__init__(name)
        self._model = LinearSVC(C=c)
        self.keys_correspondences = keys_correspondences

    def fit(self, x):
        labels_key = self.keys_correspondences["labels_key"]
        features_key = self.keys_correspondences["features_key"]

        labels = np.clip(x[labels_key], 0, 1)
        labels = -labels + 1
        self._model.fit(x[features_key], labels)

    def run(self, x):
        features_key = self.keys_correspondences["features_key"]
        scores_key = self.keys_correspondences["scores_key"]
        output_type_key = self.keys_correspondences["output_type_key"]

        x[scores_key] = self._model.decision_function(x[features_key])
        x[output_type_key] = ProcessorOutputType.LIKELIHOOD
        return x

    def to_dict(self):
        output_dict = {
            'data': np.array(pickle.dumps(self._model)),
        }

        return output_dict

    def from_dict(self, dict):
        if sys.version_info[0] < 3:
            self._model = pickle.loads(dict['data'])
        else:
            self._model = pickle.loads(dict['data'], encoding='latin1')

    def printmodel(self):
        print(self._model)

    def __str__(self):
        description = {
            'type': 'Classifier',
            'name': self.name,
            'C': self._model.C
        }
        return str(description)
