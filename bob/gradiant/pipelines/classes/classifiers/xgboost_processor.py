#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.default_keys_correspondences import DEFAULT_KEYS_CORRESPONDENCES
from bob.gradiant.pipelines.classes.processor_output_type import ProcessorOutputType
import sys
import pickle
import numpy as np
from xgboost import XGBClassifier


class XgboostProcessor(Processor):
    """
      This class encapsulates xgboost classifier wrapper class on top of sklearn.

   
      """.format(DEFAULT_KEYS_CORRESPONDENCES)

    _model = None

    def __init__(self, name='xgboost',
                 keys_correspondences=DEFAULT_KEYS_CORRESPONDENCES,
                 **kwargs):
        super(XgboostProcessor, self).__init__(name)
        self._kwargs = kwargs
        self._model = XGBClassifier(**self._kwargs)
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

        x[scores_key] = self._model.predict_proba(x[features_key])[:, 1].astype(np.float64)
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
            'type': 'XGBClassifier',
            'name': self.name,
            'kwargs': self._kwargs,
        }
        return str(description)
