#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

import h5py
import os
from bob.gradiant.pipelines.classes.processor import Processor


class PipelineLoader(Processor):
    X = None

    def __init__(self, basename_file, name='loader'):
        self.name = name
        self.basename_file = basename_file
        self.extension = '.h5'

    def fit(self, X):
        pass

    def run(self, X):
        return self.X;

    def save(self, base_path):
        pass

    def load(self, base_path):
        if base_path is None:
            raise IOError('Base path not set')

        filename = os.path.join(base_path, self.basename_file + self.extension)
        if not os.path.isfile(filename):
            raise TypeError('File (' + filename + ') does not exist')

        file_root = h5py.File(filename, 'r')
        dict_from_file = {}
        for key, value in file_root.iteritems():
            dict_from_file[key.encode("utf-8")] = value[...]
        file_root.close()
        self.X = dict_from_file

    def __str__(self):
        description = {
            'type': 'Pipeline loader',
            'name': self.name,
            'path': self.basename_file
        }
        return str(description)

    def from_dict(self, dict):
        pass

    def to_dict(self):
        pass
