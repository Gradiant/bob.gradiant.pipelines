#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain
import numpy as np
import h5py
import os
import unittest
import shutil
from bob.gradiant.pipelines.classes.storage.pipeline_saver import PipelineSaver
from bob.gradiant.pipelines.test.test_utils import TestUtils


class TestPipelineSaver(unittest.TestCase):
    def setUp(self):
        if not os.path.isdir(TestUtils.get_result_path()):
            os.mkdir(TestUtils.get_result_path())
        self.base_path = TestUtils.get_result_path()
        self.basename_file = 'test_pipeline_saver'
        self.extension = '.h5'
        self.filename = os.path.join(self.base_path, self.basename_file + self.extension)

        self.names = np.array(['grad000_real_00_00.zip', 'grad001_real_00_00.zip', 'grad002_real_00_00.zip',
                               'grad003_real_00_00.zip', 'grad004_real_00_00.zip', 'grad005_real_00_00.zip'])
        self.features = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]])
        self.labels = np.array(['real', 'real', 'attack', 'real', 'attack', 'attack'])
        self.input_dict = {'names': self.names, 'features': self.features, 'labels': self.labels}

    def tearDown(self):
        if os.path.isdir(TestUtils.get_result_path()):
            shutil.rmtree(TestUtils.get_result_path())

    def test_entry_dictionary_is_empty(self):
        empty_dict = {}
        pipeline_saver = PipelineSaver(self.basename_file)
        pipeline_saver.run(empty_dict)
        self.assertRaises(TypeError,
                          lambda: pipeline_saver.save(self.basename_file)
                          )

    def test_saved_hdf5_exists(self):
        ps = PipelineSaver(self.basename_file)
        ps.run(self.input_dict)

        ps.save(self.base_path)

        self.assertTrue(os.path.isfile(self.filename))

    def test_stored_data_is_correct(self):
        ps = PipelineSaver(self.basename_file)
        ps.run(self.input_dict)

        ps.save(self.base_path)

        file_root = h5py.File(self.filename, 'r')
        names_dset = file_root['/names']
        labels_dset = file_root['/labels']
        features_dset = file_root['/features']

        self.assertTrue(np.array_equal(names_dset[...], self.names))
        self.assertTrue(np.array_equal(features_dset[...], self.features))
        self.assertTrue(np.array_equal(labels_dset[...], self.labels))

    def test_stored_data_is_correct_if_modified_after_run(self):
        ps = PipelineSaver(self.basename_file)
        ps.run(self.input_dict)
        self.input_dict = None

        ps.save(self.base_path)

        file_root = h5py.File(self.filename, 'r')
        names_dset = file_root['/names']
        labels_dset = file_root['/labels']
        features_dset = file_root['/features']

        self.assertTrue(np.array_equal(names_dset[...], self.names))
        self.assertTrue(np.array_equal(features_dset[...], self.features))
        self.assertTrue(np.array_equal(labels_dset[...], self.labels))


if __name__ == '__main__':
    unittest.main()
