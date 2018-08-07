#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain
import os
import numpy
from PIL import Image


class TestUtils(object):
    resources_path = os.path.dirname(__file__)+'/../../../../resources'
    result_path = os.path.dirname(__file__)+'/../../../../result'

    @classmethod
    def get_resources_path(cls):
        return cls.resources_path

    @classmethod
    def get_result_path(cls):
        return cls.result_path

    @classmethod
    def get_image(cls):
        return Image.open(cls.resources_path + '/genuine/01.jpg')

    @classmethod
    def get_numpy_image(cls):
        return numpy.array(cls.get_image())

    @classmethod
    def get_synthetic_dict_image(cls, timestamp_reference=1500000000):
        dict_images = {}
        timestamp_reference = timestamp_reference
        for timestamp in range(timestamp_reference, timestamp_reference+5000, 33):
            dict_images[timestamp] = cls.get_image()
        return dict_images

