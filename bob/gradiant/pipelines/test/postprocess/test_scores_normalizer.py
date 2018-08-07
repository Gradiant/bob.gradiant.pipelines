#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

import unittest
import numpy as np
from bob.gradiant.pipelines import ScoresNormalizer


class UnitTestScoresNormalizer(unittest.TestCase):
    scores = np.array((-2., -1., 1., 2.))
    labels = np.array((0, 0, 1, 1))

    def setUp(self):
        self.X = {
            'scores': self.scores.copy(),
            'labels': self.labels.copy(),
        }

    def test_constructor_run(self):
        scores_normalizer = ScoresNormalizer()

        Y = scores_normalizer.run(self.X)

        np.testing.assert_almost_equal(Y['scores'], np.array([0, 0.25, 0.75, 1.]))
        np.testing.assert_equal(Y['labels'], np.array([0, 0, 1, 1]))


if __name__ == '__main__':
    unittest.main()
