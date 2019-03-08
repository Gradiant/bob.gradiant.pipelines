#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.default_keys_correspondences import DEFAULT_KEYS_CORRESPONDENCES
from bob.gradiant.pipelines.classes.processor_output_type import ProcessorOutputType
import numpy as np


class DummyClassifier(Processor):
    """
      This class represents a Dummy classifier. It only pass the entry values forward.

      **Parameters:**

      ``name`` : :py:class:`str`
          Name for Dummy Classifier. Default: 'dummy'.

      ``keys_correspondences`` : :py:class:`dcit`
          key for access to values on X. Default: {}.

      """.format(DEFAULT_KEYS_CORRESPONDENCES)

    _model = None

    def __init__(self, name='linear_svc',
                 keys_correspondences=DEFAULT_KEYS_CORRESPONDENCES):
        super(DummyClassifier, self).__init__(name)
        self.keys_correspondences = keys_correspondences

    def fit(self, x):
        pass

    def run(self, x):
        features_key = self.keys_correspondences["features_key"]
        scores_key = self.keys_correspondences["scores_key"]
        output_type_key = self.keys_correspondences["output_type_key"]

        x[scores_key] = x[features_key]
        x[output_type_key] = ProcessorOutputType.DISTANCE
        return x

    @staticmethod
    def to_dict():
        dict = {
            'data': np.zeros((1, 5))
        }
        return dict

    def from_dict(self, dict):
        pass

    def __str__(self):
        description = {
            'type': 'Dummy Classifier',
            'name': self.name,
        }
        return str(description)
