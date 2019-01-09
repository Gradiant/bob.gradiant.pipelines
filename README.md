# bob.gradiant.pipelines 

[![Build Status](https://travis-ci.org/Gradiant/bob.gradiant.pipelines.svg?branch=master)](https://travis-ci.org/Gradiant/bob.gradiant.pipelines)
[![Doc](http://img.shields.io/badge/docs-latest-orange.svg)](https://acostapazo.github.io/bob.gradiant.pipelines/)


[Bob](https://www.idiap.ch/software/bob/) package which defines a series of utilities that can help us to define an experiment. 
The basic processing element ([Processor](https://intranet.gradiant.org/bitbucket/projects/MBPYTHON/repos/bob.gradiant.pipelines/browse/bob/gradiant/pipelines/classes/processor.py)) represents the minimum unit for data processing. 
A [Pipeline](bob/gradiant/pad/pipelines/classes/pipeline/pipeline.py) is composed by a list of [Processor](https://intranet.gradiant.org/bitbucket/projects/MBPYTHON/repos/bob.gradiant.pipelines/browse/bob/gradiant/pipelines/classes/processor.py) objects. You can create your own pipeline by initializing it with a processor list.
You can also create your own processor(s), which must fit into the [Processor](https://intranet.gradiant.org/bitbucket/projects/MBPYTHON/repos/bob.gradiant.pipelines/browse/bob/gradiant/pipelines/classes/processor.py) interface. This repo contains several examples for [postprocess](https://intranet.gradiant.org/bitbucket/projects/MBPYTHON/repos/bob.gradiant.pipelines/browse/bob/gradiant/pipelines/classes/postprocess), [dimensionality_reduction](https://intranet.gradiant.org/bitbucket/projects/MBPYTHON/repos/bob.gradiant.pipelines/browse/bob/gradiant/pipelines/classes/dimensionality_reduction), [classifiers](https://intranet.gradiant.org/bitbucket/projects/MBPYTHON/repos/bob.gradiant.pipelines/browse/bob/gradiant/pipelines/classes/classifiers), etc.

## Docker 

The fastest way to contact the package is to use docker. 

You can download the docker image from dockerhub

~~~
docker pull acostapazo/bob.gradiant.pipelines:latest 
~~~

or build it from Dockerfile

~~~
docker build --no-cache -t acostapazo/bob.gradiant.pipelines:latest  .
~~~

To check if everything is alright you can run the ci.sh script with:

~~~
docker run -v $(pwd):/bob.gradiant.pipelines acostapazo/bob.gradiant.pipelines:latest bin/bash -c "cd bob.gradiant.pipelines; ./ci.sh"
~~~

## Installation (Manual)


1. Install conda -> https://conda.io/docs/user-guide/install/index.html

2. Create the conda env from file (environment_linux.yml)

Note: You should be inside the package directory (bob.gradiant.pad.paper.icb2019)

~~~
    conda create --name bob.gradiant.pipelines python=2.7
    conda config --env --add channels defaults
    conda config --env --add channels https://www.idiap.ch/software/bob/conda
~~~

3. Activate the environment

~~~
   source activate bob.gradiant.pipelines
~~

4. Install dependencies

~~~
    conda install gitpython h5py pillow scikit-learn mock sphinx_rtd_theme bob.extension
    pip install enum34
~~~


4. Buildout the bob package

~~~
    #You should be inside the activated conda env (bob.gradiant.pipelines)
    python bootstrap-buildout.py
    bin/buildout
~~

## Test

~~~
  bin/nosetests -v
~~~

## Clean

~~~
  python clean.py
~~~

## Coverage

~~~  
  bin/coverage run -m unittest discover
  bin/coverage html -i
  bin/coverage xml -i
~~~

Coverage result will be store on htmlcov/.

## Doc

~~~
bin/sphinx-build -b html doc/ doc/html/
~~~
