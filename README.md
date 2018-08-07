# bob.gradiant.pipelines 

[Bob](https://www.idiap.ch/software/bob/) package which defines a series of utilities that can help us to define an experiment. 
The basic processing element ([Processor](https://intranet.gradiant.org/bitbucket/projects/MBPYTHON/repos/bob.gradiant.pipelines/browse/bob/gradiant/pipelines/classes/processor.py)) represents the minimum unit for data processing. 
A [Pipeline](bob/gradiant/pad/pipelines/classes/pipeline/pipeline.py) is composed by a list of [Processor](https://intranet.gradiant.org/bitbucket/projects/MBPYTHON/repos/bob.gradiant.pipelines/browse/bob/gradiant/pipelines/classes/processor.py) objects. You can create your own pipeline by initializing it with a processor list.
You can also create your own processor(s), which must fit into the [Processor](https://intranet.gradiant.org/bitbucket/projects/MBPYTHON/repos/bob.gradiant.pipelines/browse/bob/gradiant/pipelines/classes/processor.py) interface. This repo contains several examples for [postprocess](https://intranet.gradiant.org/bitbucket/projects/MBPYTHON/repos/bob.gradiant.pipelines/browse/bob/gradiant/pipelines/classes/postprocess), [dimensionality_reduction](https://intranet.gradiant.org/bitbucket/projects/MBPYTHON/repos/bob.gradiant.pipelines/browse/bob/gradiant/pipelines/classes/dimensionality_reduction), [classifiers](https://intranet.gradiant.org/bitbucket/projects/MBPYTHON/repos/bob.gradiant.pipelines/browse/bob/gradiant/pipelines/classes/classifiers), etc.


## Environment

We strongly recommend to use [conda](https://conda.io/docs/) to manage project environment.

There is available two shared recipes to create the enviroment for this project.

*Linux*
~~~
conda env create gradiant/biometrics_py27
~~~

*Mac OSx*
~~~
conda env create gradiant/biometrics_mac_py27
~~~

If you prefer to install the environment from yaml files:

*Linux*
~~~
conda env create -f environments/biometrics_ubuntu_py27.yml
~~~

*Mac OSx*
~~~
conda env create -f environments/biometrics_mac_py27.yml
~~~


## Installation

We assume you have activate biometrics_py27 (or biometrics_mac_py27) environment 

~~~
source activate biometrics_py27
~~~

Then, you can buildout the project with:

~~~
  cd bob.gradiant.core
  python bootstrap-buildout.py
  bin/buildout
~~~

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