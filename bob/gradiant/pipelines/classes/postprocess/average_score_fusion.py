#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

from bob.gradiant.pipelines.classes.processor import Processor
import pickle
import numpy as np


class AverageScoreFusion(Processor):
    def __init__(self, name='average_score_fusion'):
        super(AverageScoreFusion, self).__init__(name)

    def fit(self, X):
        pass

    def run(self, X):
        access_ids = X['access_ids']
        labels = X['labels']
        unique_access_ids = np.unique(access_ids)

        list_average_scores = []
        list_access_id = []
        list_labels = []
        for id in unique_access_ids:
            indices = np.where(access_ids == id)
            average_score = np.average(X['scores'][indices])
            list_average_scores.append(average_score)
            list_access_id.append(access_ids[indices[0][0]])
            list_labels.append(labels[indices[0][0]])

        X['scores'] = np.array(list_average_scores)
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
        tmp_dict = {
            'data': np.array(pickle.dumps(self._model))
        }
        return tmp_dict

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
