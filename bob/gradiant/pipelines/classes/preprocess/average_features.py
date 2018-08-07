#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

from bob.gradiant.pipelines.classes.processor import Processor
import pickle
import numpy as np


class AverageFeatures(Processor):
    def __init__(self, name='average_features'):
        super(AverageFeatures, self).__init__(name)

    def fit(self, X):
        pass

    def run(self, X):
        access_ids = X['access_ids']
        labels = X['labels']
        unique_access_ids = np.unique(access_ids)

        list_average_features = []
        list_access_id = []
        list_labels = []
        for id in unique_access_ids:
            indices = np.where(access_ids == id)
            average_features = np.average(X['features'][indices], axis=0)
            list_average_features.append(average_features)
            list_access_id.append(access_ids[indices[0][0]])
            list_labels.append(labels[indices[0][0]])

        X['features'] = np.array(list_average_features)
        X['access_ids'] = np.array(list_access_id)
        X['labels'] = np.array(list_labels)
        return X

    def to_dict(self):
        return None

    def from_dict(self, dict):
        pass

    def printmodel(self):
        print(self._model)

    def to_dict(self):
        dict = {
            'data': np.array(pickle.dumps(self._model))
        }
        return dict

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
