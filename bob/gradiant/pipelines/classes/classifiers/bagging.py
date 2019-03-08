#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain
from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.default_keys_correspondences import DEFAULT_KEYS_CORRESPONDENCES
from bob.gradiant.pipelines.classes.processor_output_type import ProcessorOutputType
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import LinearSVC
import pickle
import numpy as np
import copy


class BaggingProcessor(Processor):
    def __init__(self,
                 name='bagging',
                 c=1.0,
                 keys_correspondences=DEFAULT_KEYS_CORRESPONDENCES
                 ):
        super(BaggingProcessor, self).__init__(name)
        self._model = BaggingClassifier(LinearSVC(C=c), max_samples=0.5, max_features=0.8)
        self.keys_correspondences = keys_correspondences

    def to_dict(self):
        output_dict = {
            'data': np.array(pickle.dumps(self._model)),
        }
        return output_dict

    def from_dict(self, dict):
        self._model = pickle.loads(dict['data'])

    def fit(self, x):
        labels_key = self.keys_correspondences["labels_key"]
        features_key = self.keys_correspondences["features_key"]

        labels = copy.deepcopy(x[labels_key])
        labels[labels > 0] = 1
        self._model.fit(x[features_key], labels)

    def run(self, x):
        features_key = self.keys_correspondences["features_key"]
        scores_key = self.keys_correspondences["scores_key"]
        output_type_key = self.keys_correspondences["output_type_key"]

        x[scores_key] = self._model.decision_function(x[features_key])
        x[output_type_key] = ProcessorOutputType.LIKELIHOOD
        return x

    def __str__(self):
        description = {
            'type': 'Bagging Processor',
            'name': self.name
        }
        return str(description)
