#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

import h5py
import os
import warnings
from bob.gradiant.pipelines.classes.processor import Processor


class PipelineSaver(Processor):
    X = None

    def __init__(self, basename_file, name='saver'):
        self.name = name
        self.basename_file = basename_file
        self.extension = '.h5'

    def fit(self, X):
        self.X = X
        pass

    def run(self, X):
        self.X = X
        return X

    def save(self, base_path):
        if base_path is None:
            raise IOError('Save path is not set')

        if not self.X:
            raise TypeError('Entry dictionary is empty!')

        filename = os.path.join(base_path, self.basename_file + self.extension)
        self.create_path(filename)
        if os.path.isfile(filename):
            warnings.warn('HDF5 file (' + filename + ') already exists, it will be overwritten.')
        file_root = h5py.File(filename, 'w')

        for key, value in self.X.iteritems():
            file_root.create_dataset(key, data=value)

        file_root.close()

    def load(self, x):
        pass

    def __str__(self):
        description = {
            'type': 'Pipeline saver',
            'name': self.name,
        }
        return str(description)

    def from_dict(self, dict):
        pass

    def to_dict(self):
        pass
