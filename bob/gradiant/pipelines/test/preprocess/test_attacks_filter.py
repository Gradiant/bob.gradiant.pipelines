#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

import unittest
import numpy as np
from bob.gradiant.pipelines import AttackFilter


class UnitTestAttackFilter(unittest.TestCase):
    features = np.array(([0.2, 0.2, 0.2],
                         [0.1, 0.1, 0.1],
                         [0.1, 0.1, 0.1],
                         [0.2, 0.2, 0.2],
                         [0.9, 0.9, 0.9]))
    labels = np.array((0, 1, 2, 3, 2))
    access_id = np.array((0, 0, 1, 1, 2))

    def setUp(self):
        self.X = {
            'features': self.features.copy(),
            'labels': self.labels.copy(),
            'access_ids': self.access_id.copy()
        }

    def test_constructor(self):
        attacks_filter = AttackFilter()
        np.testing.assert_equal(attacks_filter._index_filter, [1])
        attacks_filter = AttackFilter(index_filter=[1, 2])
        np.testing.assert_equal(attacks_filter._index_filter, [1, 2])

    def test_run(self):
        attacks_filter = AttackFilter(index_filter=[1, 2])
        X = attacks_filter.run(self.X)
        np.testing.assert_array_equal(X['labels'], np.array((0, 1, 2, 2)))
        filter_features = np.array(([0.2, 0.2, 0.2],
                                    [0.1, 0.1, 0.1],
                                    [0.1, 0.1, 0.1],
                                    [0.9, 0.9, 0.9]))
        np.testing.assert_array_equal(X['features'], filter_features)

    def test_attack_not_in_list(self):
        attacks_filter = AttackFilter(index_filter=[6])
        X = attacks_filter.run(self.X)
        np.testing.assert_array_equal(X['labels'], np.array([0]))
        np.testing.assert_array_equal(X['features'], np.array([[0.2, 0.2, 0.2]]))


if __name__ == '__main__':
    unittest.main()
