#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

import os
import h5py
import numpy as np
from bob.gradiant.pipelines.classes.processor import Processor

try:
    basestring
except NameError:
    basestring = str


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
        for key, value in file_root.items():
            if self._is_list_of_strings(value[...]):
                dict_from_file[str(key)] = [x.decode('utf-8') for x in value[...]]

            elif self._is_numpy_array_of_bytestrings(value[...]):
                dict_from_file[str(key)] = value[...].astype('U')

            else:
                dict_from_file[str(key)] = value[...]

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
        elif np.issubdtype(arr.dtype, np.dtype('S')):
            return True
        else:
            return False
