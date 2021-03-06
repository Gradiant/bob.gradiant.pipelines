#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.default_keys_correspondences import DEFAULT_KEYS_CORRESPONDENCES
import pickle
import numpy as np


class AverageFeatures(Processor):
    def __init__(self,
                 name='average_features',
                 keys_correspondences=DEFAULT_KEYS_CORRESPONDENCES):
        super(AverageFeatures, self).__init__(name)
        self.keys_correspondences = keys_correspondences

    def fit(self, x):
        pass

    def run(self, x):
        labels_key = self.keys_correspondences["labels_key"]
        features_key = self.keys_correspondences["features_key"]
        access_ids_key = self.keys_correspondences["access_ids_key"]

        access_ids = x[access_ids_key]
        labels = x[labels_key]
        unique_access_ids = np.unique(access_ids)

        list_average_features = []
        list_access_id = []
        list_labels = []
        for access_id in unique_access_ids:
            indices = np.where(access_ids == access_id)
            average_features = np.average(x[features_key][indices], axis=0)
            list_average_features.append(average_features)
            list_access_id.append(access_ids[indices[0][0]])
            list_labels.append(labels[indices[0][0]])

        x[features_key] = np.array(list_average_features)
        x[access_ids_key] = np.array(list_access_id)
        x[labels_key] = np.array(list_labels)
        return x

    def to_dict(self):
        return None

    def from_dict(self, dict):
        pass

    def printmodel(self):
        print(self._model)

    def save(self, base_path):
        pass

    def load(self, base_path):
        pass

    def __str__(self):
        description = {
            'type': 'average score fusion',
            'name': self.name
        }
        return str(description)
