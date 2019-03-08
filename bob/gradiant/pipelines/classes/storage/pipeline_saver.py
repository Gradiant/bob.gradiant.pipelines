#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

import os
import h5py
import warnings
import numpy as np
from bob.gradiant.pipelines.classes.processor import Processor

try:
    basestring
except NameError:
    basestring = str


class PipelineSaver(Processor):
    X = None

    def __init__(self, basename_file, name='saver'):
        self.name = name
        self.basename_file = basename_file
        self.extension = '.h5'

    def fit(self, x):
        self.X = x
        pass

    def run(self, x):
        self.X = x
        return x

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

        for key, value in self.X.items():
            if self._is_list_of_strings(value):
                file_root.create_dataset(key, data=np.array(value, dtype='S'))
            elif self._is_numpy_array_of_bytestrings(value):
                file_root.create_dataset(key, data=value.astype('S'))
            else:
                file_root.create_dataset(key, data=value)

        file_root.close()

    @staticmethod
    def _is_list_of_strings(lst):
        if not isinstance(lst, list):
            return False
        else:
            return bool(lst) and not isinstance(lst, basestring) and all(isinstance(elem, basestring) for elem in lst)

    @staticmethod
    def _is_numpy_array_of_bytestrings(arr):
        if not isinstance(arr, np.ndarray):
            return False
        elif np.issubdtype(arr.dtype, np.str_):
            return True
        else:
            return False

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
