#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain
from abc import ABCMeta, abstractmethod
import os.path
import h5py


class Processor(object):
    __metaclass__ = ABCMeta
    name = None

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def fit(self, X):
        raise NotImplementedError

    @abstractmethod
    def run(self, X):
        raise NotImplementedError

    @abstractmethod
    def to_dict(self):
        raise NotImplementedError

    @abstractmethod
    def from_dict(self, dict):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    def save(self, base_path):
        output_file_path = self.__compute_file_name(base_path)
        self.create_path(output_file_path)
        dict = self.to_dict()
        h5f = h5py.File(output_file_path, 'w')

        for k in dict:
            h5f.create_dataset(k, data=dict[k])

        h5f.close()

    def load(self, base_path):
        file_path = self.__compute_file_name(base_path)
        if not os.path.exists(file_path):
            print(file_path)
            print('File {} not found'.format(file_path))
            raise IOError()

        dict = {}
        h5f = h5py.File(file_path, 'r')

        for k in h5f:
            dict[k] = h5f[k].value
        self.from_dict(dict)

    def fit_run(self, X):
        self.fit(X)
        return self.run(X)

    def __compute_file_name(self, base_path):
        return os.path.join(base_path, 'processors', self.name + '.h5')

    @staticmethod
    def create_path(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
