#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

import unittest
import numpy as np
from bob.gradiant.pipelines import AverageFeatures


class UnitTestAverageFeatures(unittest.TestCase):
    features = np.array(([0.2, 0.2, 0.2],
                         [0.1, 0.1, 0.1],
                         [0.1, 0.1, 0.1],
                         [0.2, 0.2, 0.2],
                         [0.9, 0.9, 0.9]))
    labels = np.array((0, 0, 0, 0, 1))
    access_id = np.array((0, 0, 1, 1, 2))

    def setUp(self):
        self.X = {
            'features': self.features.copy(),
            'labels': self.labels.copy(),
            'access_ids': self.access_id.copy()
        }

    def test_run(self):
        average_features = AverageFeatures()

        Y = average_features.run(self.X)

        np.testing.assert_almost_equal(Y['features'], np.array(([0.15, 0.15, 0.15],
                                                                [0.15, 0.15, 0.15],
                                                                [0.9, 0.9, 0.9])))
        np.testing.assert_equal(Y['labels'], np.array([0, 0, 1]))
        np.testing.assert_equal(Y['access_ids'], np.array([0, 1, 2]))


if __name__ == '__main__':
    unittest.main()
