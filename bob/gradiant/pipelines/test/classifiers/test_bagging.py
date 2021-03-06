#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

import copy
import os.path
import pickle
import unittest

import h5py
import numpy as np
from mock import MagicMock, patch
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import LinearSVC

from bob.gradiant.pipelines import BaggingProcessor
from bob.gradiant.pipelines.test.test_utils import TestUtils


class MyDataset():
    value = None

    def __init__(self, value):
        self.value = value


class UnitTestBagging(unittest.TestCase):
    features = np.array(((0.3056649, 0.16551992, 0.81751581, 0.13783114, 0.88282389),
                         (0.43597483, 0.47153349, 0.48909926, 0.06524072, 0.54416104),
                         (0.32332448, 0.92072384, 0.42503036, 0.77241727, 0.66704238),
                         (0.31197222, 0.15767281, 0.37699822, 0.06159546, 0.54474634),
                         (0.28018542, 0.00218424, 0.92349832, 0.57061759, 0.21193483),
                         (0.16879066, 0.25092238, 0.01884595, 0.66949375, 0.72722846),
                         ))

    labels = np.array((1, 0, 2, 0, 1, 0))
    X = {
        'features': features.copy(),
        'labels': labels.copy()
    }

    scores = np.array((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))

    base_path = TestUtils.get_result_path() + '/adaboost_tests'

    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    h5f = h5py.File(base_path + '/bag_test.h5', 'w')

    @patch('sklearn.ensemble.BaggingClassifier.__init__', MagicMock(return_value=None))
    def test_constructor_calls_sklearn_constructor(self):
        BaggingProcessor(c=0.94)
        BaggingClassifier.__init__.assert_called_once()

    @patch('sklearn.ensemble.BaggingClassifier.fit', MagicMock())
    def test_fit_calls_sklearn_fit(self):
        bag = BaggingProcessor()
        bag.fit(self.X)

        labels = copy.deepcopy(self.X['labels'])
        expected_labels = copy.deepcopy(labels)
        expected_labels[expected_labels > 0] = 1

        np.testing.assert_array_equal(BaggingClassifier.fit.call_args[0][0], self.X['features'])
        np.testing.assert_array_equal(BaggingClassifier.fit.call_args[0][1], expected_labels)

    @patch('sklearn.ensemble.BaggingClassifier.decision_function', MagicMock(return_value=scores))
    def test_transform_calls_sklearn_transform(self):
        features = self.X['features'].copy()
        bag = BaggingProcessor()

        bag.run(self.X)

        np.testing.assert_array_equal(BaggingClassifier.decision_function.call_args[0][0], features)

    @patch('pickle.dumps', MagicMock(return_value='Model dump'))
    @patch('h5py.File', MagicMock(return_value=h5f))
    @patch('h5py.File.create_dataset', MagicMock())
    def test_save_model(self):
        bag = BaggingProcessor(name='TestBag')

        bag.save(self.base_path)

        h5py.File.assert_called_once_with(os.path.join(self.base_path, 'processors/TestBag.h5'), 'w')
        self.h5f.create_dataset.assert_called_once_with('data', data=np.array('Model dump'))

    dataset = {
        'data': MyDataset('Model')
    }

    @patch('pickle.loads', MagicMock(return_value='Model'))
    @patch('h5py.File', MagicMock(return_value=dataset))
    @patch('os.path.exists', MagicMock(return_value=True))
    def test_load_model(self):
        bag = BaggingProcessor(name='TestBag')
        bag.load(self.base_path)

        h5py.File.assert_called_once_with(os.path.join(self.base_path, 'processors/TestBag.h5'), 'r')
        pickle.loads.assert_called_once_with('Model')
        self.assertEquals('Model', bag._model)

    def test_describe(self):
        description = BaggingProcessor(c=0.75).__str__()

        self.assertEquals(description, '{\'type\': \'Bagging Processor\', \'name\': \'bagging\'}')


if __name__ == '__main__':
    unittest.main()