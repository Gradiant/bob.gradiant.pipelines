#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain
import copy
from bob.gradiant.pipelines.classes.processor import Processor

try:
    basestring
except NameError:
    basestring = str


class Pipeline:
    processor_list = []

    def __init__(self, name, list):
        if not isinstance(name, basestring):
            raise TypeError("name must be a basestring")
        self.name = name
        for p in list:
            if not isinstance(p, Processor):
                raise TypeError("All pipeline elements must a Processor")
        self.processor_list = list

    def fit(self, X):
        p_X = copy.deepcopy(X)
        for p in self.processor_list[:-1]:
            p_X = p.fit_run(p_X)
        self.processor_list[-1].fit(p_X)

    def run(self, X):
        p_X = X
        for p in self.processor_list:
            p_X = p.run(p_X)
        return p_X

    def fit_run(self, X):
        p_X = X
        for p in self.processor_list:
            p_X = p.fit_run(p_X)
        return p_X

    def save(self, base_path):
        for p in self.processor_list:
            p.save(base_path)

    def load(self, base_path):
        for p in self.processor_list:
            p.load(base_path)

    def __str__(self):
        description = self.processor_list[0].__str__()
        for p in self.processor_list[1:]:
            description += ' -> ' + p.__str__()
        return '{} : {}'.format(self.__class__.__name__, description)
