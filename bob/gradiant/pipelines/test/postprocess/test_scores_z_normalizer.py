#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

import unittest
import numpy as np
from bob.gradiant.pipelines import ScoresZNormalizer


class UnitTestScoresZNormalizer(unittest.TestCase):
    scores = np.array((-2., -1., 1., 2.))
    labels = np.array((0, 0, 1, 1))

    def setUp(self):
        self.x = {
            'scores': self.scores.copy(),
            'labels': self.labels.copy(),
        }

    def test_constructor_fit(self):
        scores_normalizer = ScoresZNormalizer()

        y = scores_normalizer.fit(self.x)

        self.assertEqual(scores_normalizer.mean, 0)
        self.assertEqual(scores_normalizer.std, 1.5811388300841898)

    def test_constructor_run(self):
        scores_normalizer = ScoresZNormalizer()

        y = scores_normalizer.run(self.x)
        self.assertTrue(np.array_equal(y['scores'], np.array([0., 0., 1., 1.])))


if __name__ == '__main__':
    unittest.main()
