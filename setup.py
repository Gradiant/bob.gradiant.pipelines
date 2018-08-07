#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017+ Gradiant, Vigo, Spain

from setuptools import setup, find_packages
from version import *

setup(
    name='bob.gradiant.pipelines',
    version=get_version(),
    description='Template for gradiant python packages',
    url='http://pypi.python.org/pypi/template-gradiant-python',
    license='GPLv3',
    author='Biometrics Team (Gradiant)',
    author_email='biometrics.support@gradiant.org',
    long_description=open('README.md').read(),
    keywords='template gradiant',

    # This line is required for any distutils based packaging.
    packages=find_packages(),
    include_package_data=True,
    zip_safe=True,

    install_requires=[
      "setuptools",
    ],

    entry_points={
      'console_scripts': [
        'pipeline_example.py = bob.gradiant.pipelines.scripts.pipeline_example:main',
        ],
    },
)
