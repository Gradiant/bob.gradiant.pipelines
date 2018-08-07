#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain
import unittest
from mock import MagicMock, patch, call
from bob.gradiant.pipelines import Processor, Pipeline


class Concreter(Processor):
    def __init__(self):
        super(Concreter, self).__init__('Concreter')

    def from_dict(self, dict):
        pass

    def to_dict(self):
        pass

    def save(self, base_path):
        pass

    def run(self, X):
        pass

    def fit(self, X):
        pass

    def __str__(self):
        pass


class UnitTestPipeline(unittest.TestCase):
    base_path = 'base/path'

    def test_no_name_in_pipeline_raises_type_error(self):
        self.assertRaises(TypeError,
                          lambda: Pipeline(None,[])
                          )


    @patch('bob.gradiant.pipelines.test.test_pipeline.Concreter.__init__',
           MagicMock())
    def test_no_preprocessor_in_pipeline_raises_type_error(self):
        self.assertRaises(TypeError,
                          lambda: Pipeline('name_pipeline',[
                              Concreter(),
                              str("I'm not a preprocessor"),
                          ])
                          )

    @patch('bob.gradiant.pipelines.test.test_pipeline.Concreter.fit',
           MagicMock())
    def test_fit_is_called_for_single_processor(self):
        pipeline = Pipeline('name_pipeline',[
            Concreter(),
        ])
        X = (0, 1)

        pipeline.fit(X)

        Concreter.fit.assert_called_once_with(X)

    @patch('bob.gradiant.pipelines.test.test_pipeline.Concreter.run',
           MagicMock(return_value=(1, 0)))
    def test_run_is_called_for_single_processor(self):
        pipeline = Pipeline('name_pipeline',[
            Concreter(),
        ])
        X = (0, 1)

        Y = pipeline.run(X)

        Concreter.run.assert_called_once_with(X)
        self.assertEqual(Y, (1, 0))

    @patch('bob.gradiant.pipelines.test.test_pipeline.Concreter.fit_run',
           MagicMock(return_value=(1, 0)))
    def test_fit_run_is_called_for_single_processor(self):
        pipeline = Pipeline('name_pipeline',[
            Concreter(),
        ])
        X = (0, 1)

        Y = pipeline.fit_run(X)

        Concreter.fit_run.assert_called_once_with(X)
        self.assertEqual(Y, (1, 0))

    @patch('bob.gradiant.pipelines.test.test_pipeline.Concreter.fit',
           MagicMock())
    @patch('bob.gradiant.pipelines.test.test_pipeline.Concreter.fit_run',
           MagicMock(return_value=(1, 0)))
    def test_fit_call_for_several_processors(self):
        pipeline = Pipeline('name_pipeline',[
            Concreter(),
            Concreter(),
        ])
        X = [0, 1]

        pipeline.fit(X)

        Concreter.fit_run.assert_called_once_with(X)
        Concreter.fit.assert_called_once_with((1, 0))

    @patch('bob.gradiant.pipelines.test.test_pipeline.Concreter.run',
           MagicMock(side_effect=[(1, 0), (2, 1)]))
    def test_run_call_for_several_processors(self):
        pipeline = Pipeline('name_pipeline',[
            Concreter(),
            Concreter(),
        ])
        X = (0, 1)

        Y = pipeline.run(X)

        Concreter.run.assert_has_calls([call(X),
                                        call((1, 0))],
                                       any_order=False)
        self.assertEqual(Y, (2, 1))

    @patch('bob.gradiant.pipelines.test.test_pipeline.Concreter.fit',
           MagicMock())
    @patch('bob.gradiant.pipelines.test.test_pipeline.Concreter.run',
           MagicMock(side_effect=[(1, 0), (2, 1)]))
    def test_fit_run_call_for_several_processors(self):
        pipeline = Pipeline('name_pipeline',[
            Concreter(),
            Concreter(),
        ])
        X = (0, 1)

        Y = pipeline.fit_run(X)

        Concreter.fit.assert_has_calls([call(X),
                                        call((1, 0))])
        Concreter.run.assert_has_calls([call(X),
                                        call((1, 0))])
        self.assertEqual(Y, (2, 1))

    @patch('bob.gradiant.pipelines.test.test_pipeline.Concreter.save',
           MagicMock())
    def test_save_is_called_for_single_processor(self):
        pipeline = Pipeline('name_pipeline',[
            Concreter(),
        ])

        pipeline.save(self.base_path)

        Concreter.save.assert_called_once_with(self.base_path)

    @patch('bob.gradiant.pipelines.test.test_pipeline.Concreter.save',
           MagicMock())
    def test_save_is_called_for_several_processors(self):
        pipeline = Pipeline('name_pipeline',[
            Concreter(),
            Concreter(),
        ])

        pipeline.save(self.base_path)

        Concreter.save.assert_has_calls([call(self.base_path),
                                         call(self.base_path)])

    @patch('bob.gradiant.pipelines.test.test_pipeline.Concreter.__str__',
           MagicMock(return_value='{\'type\': \'concreter\'}'))
    def test_describe_is_called_for_single_processor(self):
        pipeline = Pipeline('name_pipeline',[
            Concreter(),
        ])

        description = pipeline.__str__()
        Concreter.__str__.assert_called_once()
        self.assertEquals('Pipeline : {\'type\': \'concreter\'}',
                          description)

    @patch('bob.gradiant.pipelines.test.test_pipeline.Concreter.__str__',
           MagicMock(side_effect=['{\'type\': \'concreter1\'}', '{\'type\': \'concreter2\'}']))
    def test_describe_is_called_for_several_processors(self):
        pipeline = Pipeline('name_pipeline',[
            Concreter(),
            Concreter(),
        ])
        description = pipeline.__str__()
        self.assertEquals('Pipeline : {\'type\': \'concreter1\'} -> {\'type\': \'concreter2\'}',
                          description)


if __name__ == '__main__':
    unittest.main()
