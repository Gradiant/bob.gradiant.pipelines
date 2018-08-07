#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain
import numpy as np
import os
import unittest
import shutil
from bob.gradiant.pipelines.classes.storage.pipeline_saver import PipelineSaver
from bob.gradiant.pipelines.classes.storage.pipeline_loader import PipelineLoader
from bob.gradiant.pipelines.test.test_utils import TestUtils


class TestPipelineLoader(unittest.TestCase):

    def _assert_my_dict(self, ref_dic, assert_dict):

        self.assertListEqual(sorted(ref_dic.keys()),sorted(assert_dict.keys()))

        for key in ref_dic.keys():
            ref_v = ref_dic[key]
            assert_v = assert_dict[key]

            np.testing.assert_array_equal(ref_v, assert_v, "Value(s) in %s mismatch" % key)

    def setUp(self):
        if not os.path.isdir(TestUtils.get_result_path()):
            os.makedirs(TestUtils.get_result_path())
        self.base_path = TestUtils.get_result_path()
        self.basename_file = 'test_pipeline_loader'
        self.extension = '.h5'
        self.filename = os.path.join(self.base_path, self.basename_file + self.extension)

        self.names = np.array(['grad000_real_00_00.zip', 'grad001_real_00_00.zip', 'grad002_real_00_00.zip',
                          'grad003_real_00_00.zip', 'grad004_real_00_00.zip', 'grad005_real_00_00.zip'])
        self.features = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]])
        self.labels = np.array(['real', 'real', 'attack', 'real', 'attack', 'attack'])
        self.input_dict = {'names': self.names, 'features': self.features, 'labels': self.labels}

        ps = PipelineSaver(self.basename_file)
        ps.run(self.input_dict)
        ps.save(self.base_path)

    def tearDown(self):
        if os.path.isdir(TestUtils.get_result_path()):
            shutil.rmtree(TestUtils.get_result_path())

    def test_input_file_does_not_exists(self):
        pl = PipelineLoader('fake_basename')
        self.assertRaises(TypeError,
                          lambda: pl.load(self.base_path)
                          )

    def test_output_dict_is_correct(self):
        pl = PipelineLoader(self.basename_file)
        pl.load(self.base_path)

        self._assert_my_dict(self.input_dict, pl.X)

if __name__ == '__main__':
    unittest.main()
