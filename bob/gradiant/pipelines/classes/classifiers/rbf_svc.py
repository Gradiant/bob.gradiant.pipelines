#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.default_keys_correspondences import DEFAULT_KEYS_CORRESPONDENCES
from bob.gradiant.pipelines.classes.processor_output_type import ProcessorOutputType
from sklearn.svm import SVC
import sys
import pickle
import numpy as np
from dill import dill
dill._reverse_typemap['ObjectType'] = object # Make it compatible with Objects serialized with pickle in Python 2


class RbfSvc(Processor):
    """
      This class encapsulates sklearn.svm.SVC.

      **Parameters:**

      ``name`` : :py:class:`str`
          Name for RbfSvc. Default: 'rbf_svc'.

      ``C`` : :py:class:`float`
          Cost function. Default: 1.0.

      ``gamma`` : :py:class:`str`
          gamma function. Default: auto.

       ``keys_correspondences`` : :py:class:`dcit`
          key for access to values on X. Default: {}.
      """.format(DEFAULT_KEYS_CORRESPONDENCES)

    _model = None

    def __init__(self, name='rbf_svc',
                 c=1.0,
                 gamma='auto',
                 keys_correspondences=DEFAULT_KEYS_CORRESPONDENCES):
        super(RbfSvc, self).__init__(name)
        self._model = SVC(C=c, gamma=gamma)
        self.keys_correspondences = keys_correspondences

    def fit(self, x):
        labels_key = self.keys_correspondences["labels_key"]
        features_key = self.keys_correspondences["features_key"]
        labels = x[labels_key].ravel()
        labels = np.clip(labels, 0, 1)
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
            'C': self._model.C,
            'gamma': self._model.gamma
        }
        return str(description)
