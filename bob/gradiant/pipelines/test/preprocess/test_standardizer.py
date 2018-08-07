#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain
import os
import pickle
import unittest

import h5py
import numpy as np
from mock import MagicMock, patch
from sklearn.preprocessing import StandardScaler

from bob.gradiant.pipelines import Standardizer
from bob.gradiant.pipelines.test.test_utils import TestUtils


class MyDataset():
    value = None

    def __init__(self, value):
        self.value = value


class UnitTestStandardizer(unittest.TestCase):
    features = 0.9 * np.random.rand(10, 5) + 5.7

    base_path = TestUtils.get_result_path() + '/standardizer_tests'

    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    h5f = h5py.File(base_path + '/standardizer_test.h5', 'w')

    h5py_dataset = {
        'data': MyDataset('Model')
    }

    def setUp(self):
        self.X = {
            'features': self.features.copy()
        }

    @patch('sklearn.preprocessing.StandardScaler.__init__', MagicMock(return_value=None))
    def test_constructor_calls_sklearn_constructor(self):
        Standardizer()
        StandardScaler.__init__.assert_called_once()

    @patch('sklearn.preprocessing.StandardScaler.fit', MagicMock(return_value=None))
    def test_fit_method(self):
        standardizer = Standardizer()

        standardizer.fit(self.X)

        StandardScaler.fit.assert_called_once_with(self.X['features'])
        self.assertNotEquals(standardizer._model, None)

    @patch('sklearn.preprocessing.StandardScaler.fit', MagicMock(return_value=None))
    @patch('sklearn.preprocessing.StandardScaler.transform', MagicMock(return_value=np.array([[0, 1], [2, 3]])))
    def test_fit_run(self):
        standardizer = Standardizer()

        Y = standardizer.fit_run(self.X)

        np.testing.assert_array_equal(StandardScaler.fit.call_args[0][0], self.features)
        np.testing.assert_array_equal(StandardScaler.transform.call_args[0][0], self.features)
        self.assertNotEquals(standardizer._model, None)
        np.testing.assert_equal(Y['features'], [[0, 1], [2, 3]])

    @patch('sklearn.preprocessing.StandardScaler.transform', MagicMock(return_value=np.array([[0, 1], [2, 3]])))
    def test_run(self):
        standardizer = Standardizer()
        Y = standardizer.run(self.X)
        np.testing.assert_array_equal(StandardScaler.transform.call_args[0][0], self.features)
        np.testing.assert_equal(Y['features'], [[0, 1], [2, 3]])

    @patch('pickle.dumps', MagicMock(return_value='Model dump'))
    @patch('h5py.File', MagicMock(return_value=h5f))
    @patch('h5py.File.create_dataset', MagicMock())
    def test_save_model(self):
        standardizer = Standardizer(name='TestStandardizer')

        standardizer.save(self.base_path)

        h5py.File.assert_called_once_with(os.path.join(self.base_path, 'processors/TestStandardizer.h5'), 'w')
        self.h5f.create_dataset.assert_called_once_with('data', data=np.array('Model dump'))

    @patch('pickle.loads', MagicMock(return_value='Model'))
    @patch('h5py.File', MagicMock(return_value=h5py_dataset))
    @patch('os.path.exists', MagicMock(return_value=True))
    def test_load_model(self):
        standardizer = Standardizer(name='TestStandardizer')
        standardizer.load(self.base_path)

        h5py.File.assert_called_once_with(os.path.join(self.base_path, 'processors/TestStandardizer.h5'), 'r')
        pickle.loads.assert_called_once_with('Model')
        self.assertEquals('Model', standardizer._model)

    def test_describe(self):
        description = Standardizer(name='test_standardizer').__str__()
        self.assertEquals(description, '{\'type\': \'Standard Scaler\', \'name\': \'test_standardizer\'}')


if __name__ == '__main__':
    unittest.main()
