#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

import unittest
import numpy as np
from bob.gradiant.pipelines import AverageScoreFusion


class UnitTestAverageScoreFusion(unittest.TestCase):
    scores = np.array((0.1, 0.2, 0.3, 0.4, 0.8))
    labels = np.array((0, 0, 0, 0, 1))
    access_id = np.array((0, 0, 1, 1, 2))

    def setUp(self):
        self.X = {
            'scores': self.scores.copy(),
            'labels': self.labels.copy(),
            'access_ids': self.access_id.copy()
        }

    def test_constructor_run(self):
        average_score_fusion = AverageScoreFusion()

        Y = average_score_fusion.run(self.X)

        np.testing.assert_almost_equal(Y['scores'], np.array([0.15, 0.35, 0.8]))
        np.testing.assert_equal(Y['labels'], np.array([0, 0, 1]))
        np.testing.assert_equal(Y['access_ids'], np.array([0, 1, 2]))


if __name__ == '__main__':
    unittest.main()
