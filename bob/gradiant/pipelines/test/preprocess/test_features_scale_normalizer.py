#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

import unittest
import numpy as np
from mock import MagicMock, patch
from sklearn.preprocessing import StandardScaler
from bob.gradiant.pipelines import FeaturesScaleNormalizer


class UnitTestFeaturesScaleNormalizer(unittest.TestCase):
    transformed_data = np.array(([[-2.44948974, -1.22474487, -0.26726124]]))
    labels = np.array((0, 1))

    X = {'Train':
        {
            'features': np.array([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]]),
            'labels': labels.copy()
        },
        'Test':
            {
                'features': np.array([[-1., -1., 0.]]),
                'labels': labels.copy()
            }
    }

    @patch('sklearn.preprocessing.StandardScaler.__init__', MagicMock(return_value=None))
    def test_constructor_calls_sklearn_constructor(self):
        FeaturesScaleNormalizer()
        StandardScaler.__init__.assert_called_once_with()

    @patch('sklearn.preprocessing.StandardScaler.fit', MagicMock())
    def test_fit_calls_sklearn_fit(self):
        feature_scale_normalzier = FeaturesScaleNormalizer()
        feature_scale_normalzier.fit(self.X['Train'])
        StandardScaler.fit.assert_called_once_with(self.X['Train']['features'])

    def test_fit_run(self):
        features_scale_normalizer = FeaturesScaleNormalizer()

        features_scale_normalizer.fit(self.X['Train'])
        Y = features_scale_normalizer.run(self.X['Test'])

        np.testing.assert_almost_equal(Y['features'], self.transformed_data)
        np.testing.assert_equal(Y['labels'], np.array([0, 1]))


if __name__ == '__main__':
    unittest.main()
