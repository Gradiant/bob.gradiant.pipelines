#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

import sys
import os.path
import pickle
import unittest

import h5py
import numpy as np
from mock import MagicMock, patch
from sklearn.decomposition import PCA

from bob.gradiant.pipelines import Pca
from bob.gradiant.pipelines.test.test_utils import TestUtils


class MyDataset():
    value = None

    def __init__(self, value):
        self.value = value


class UnitTestPca(unittest.TestCase):
    features = np.array(((0.3056649, 0.16551992, 0.81751581, 0.13783114, 0.88282389),
                         (0.43597483, 0.47153349, 0.48909926, 0.06524072, 0.54416104),
                         (0.32332448, 0.92072384, 0.42503036, 0.77241727, 0.66704238),
                         (0.31197222, 0.15767281, 0.37699822, 0.06159546, 0.54474634),
                         (0.28018542, 0.00218424, 0.92349832, 0.57061759, 0.21193483),
                         (0.16879066, 0.25092238, 0.01884595, 0.66949375, 0.72722846),
                         ))
    X = {
        'features': features.copy()
    }

    transformed_data = np.array(
        ((0.1, 0.2),
         (0.3, 0.4),
         (0.5, 0.6),)
    )

    base_path = TestUtils.get_result_path() + '/pca_tests'

    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    h5f = h5py.File(base_path + '/pca_test.h5', 'w')

    h5py_dataset = {
        'data': MyDataset('Model')
    }

    @patch('sklearn.decomposition.PCA.__init__', MagicMock(return_value=None))
    def test_constructor_calls_sklearn_constructor(self):
        Pca(n_components=0.47)
        PCA.__init__.assert_called_once_with(n_components=0.47)

    @patch('sklearn.decomposition.PCA.fit', MagicMock())
    def test_fit_calls_sklearn_fit(self):
        pca = Pca()
        pca.fit(self.X)

        PCA.fit.assert_called_once_with(self.X['features'])

    @patch('sklearn.decomposition.PCA.transform', MagicMock(return_value=transformed_data))
    def test_transform_calls_sklearn_transform(self):
        features = self.X['features'].copy()
        pca = Pca()

        pca.run(self.X)

        np.testing.assert_array_equal(PCA.transform.call_args[0][0], features)

    @patch('pickle.dumps', MagicMock(return_value='Model dump'))
    @patch('h5py.File', MagicMock(return_value=h5f))
    @patch('h5py.File.create_dataset', MagicMock())
    def test_save_model(self):
        pca = Pca(name='TestPca')

        pca.save(self.base_path)

        h5py.File.assert_called_once_with(os.path.join(self.base_path, 'processors/TestPca.h5'), 'w')
        self.h5f.create_dataset.assert_called_once_with('data', data=np.array('Model dump'))

    @patch('pickle.loads', MagicMock(return_value='Model'))
    @patch('h5py.File', MagicMock(return_value=h5py_dataset))
    @patch('os.path.exists', MagicMock(return_value=True))
    def test_load_model(self):
        pca = Pca(name='TestPca')
        pca.load(self.base_path)

        h5py.File.assert_called_once_with(os.path.join(self.base_path, 'processors/TestPca.h5'), 'r')
        if sys.version_info[0] < 3:
            pickle.loads.assert_called_once_with('Model')
        else:
            pickle.loads.assert_called_once_with('Model', encoding='latin1')
        self.assertEquals('Model', pca._model)

    def test_describe(self):
        description = Pca(n_components=0.75).__str__()
        self.assertEquals(description, '{\'type\': \'Dimensionality reduction\', \'name\': \'pca\', \'n_components\': 0.75}')


if __name__ == '__main__':
    unittest.main()
