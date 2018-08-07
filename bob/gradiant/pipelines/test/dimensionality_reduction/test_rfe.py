#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

import unittest
import numpy as np
from bob.gradiant.pipelines import RecursiveFeatureElimination, Pipeline, PipelineLoader, PipelineSaver


class UnitTestRecursiveFeatureElimination(unittest.TestCase):
    features = np.array(([0.2, 0.2, 0.2],
                         [0.1, 0.1, 0.1],
                         [0.1, 0.1, 0.1],
                         [0.2, 0.2, 0.2],
                         [0.1, 0.2, 0.9],
                         [0.2, 0.1, 0.8]))
    labels = np.array((0, 0, 0, 0, 1, 1))
    access_id = np.array((0, 0, 1, 1, 2, 2))

    def setUp(self):
        self.X = {'features': self.features.copy(), 'labels': self.labels.copy(), 'access_ids': self.access_id.copy()}

    def test_run(self):
        reduced_features = RecursiveFeatureElimination(n_features=1)
        reduced_features.fit(self.X)

        reduced_features.save('/tmp/test_rfe')

        loaded_rfe = RecursiveFeatureElimination()
        loaded_rfe.load('/tmp/test_rfe')
        Y = loaded_rfe.run(self.X)

        np.testing.assert_almost_equal(Y['features'], np.array(([[0.2],
                                                                 [0.1],
                                                                 [0.1],
                                                                 [0.2],
                                                                 [0.9],
                                                                 [0.8]])))
        np.testing.assert_equal(Y['labels'], np.array([0, 0, 0, 0, 1, 1]))
        np.testing.assert_equal(Y['access_ids'], np.array([0, 0, 1, 1, 2, 2]))
        np.testing.assert_equal(reduced_features._model.ranking_[2], 1)

    def test_run_pipeline(self):
        reduced_features = RecursiveFeatureElimination(n_features=1)
        pipeline = Pipeline('name_pipeline',[reduced_features])
        pipeline.fit(self.X)

        pipeline.save('/tmp/test_rfe_pipeline')

        loaded_rfe = RecursiveFeatureElimination()
        loaded_rfe.load('/tmp/test_rfe_pipeline')
        loaded_pipeline = Pipeline('name_pipeline',[loaded_rfe])
        Y = loaded_pipeline.run(self.X)
        np.testing.assert_almost_equal(Y['features'], np.array(([[0.2],
                                                                 [0.1],
                                                                 [0.1],
                                                                 [0.2],
                                                                 [0.9],
                                                                 [0.8]])))
        np.testing.assert_equal(Y['labels'], np.array([0, 0, 0, 0, 1, 1]))
        np.testing.assert_equal(Y['access_ids'], np.array([0, 0, 1, 1, 2, 2]))

    def test_run_pipeline_saver(self):
        reduced_features = RecursiveFeatureElimination(n_features=1)
        pipeline = Pipeline('name_pipeline',[reduced_features, PipelineSaver('dim_reduction', 'dim_reduction')])
        pipeline.fit(self.X)
        pipeline.save('/tmp/test_rfe_pipeline_saver')

        loaded_pipeline = Pipeline('name_pipeline',[PipelineLoader('dim_reduction', 'dim_reduction')])
        loaded_pipeline.load('/tmp/test_rfe_pipeline_saver')
        Y = loaded_pipeline.run(self.X)
        np.testing.assert_equal(Y['features'], np.array(([[0.2],
                                                          [0.1],
                                                          [0.1],
                                                          [0.2],
                                                          [0.9],
                                                          [0.8]])))
        np.testing.assert_equal(Y['labels'], np.array([0, 0, 0, 0, 1, 1]))
        np.testing.assert_equal(Y['access_ids'], np.array([0, 0, 1, 1, 2, 2]))
        np.testing.assert_equal(Y['indices'], np.array([2]))  # indices of the selected features


if __name__ == '__main__':
    unittest.main()
