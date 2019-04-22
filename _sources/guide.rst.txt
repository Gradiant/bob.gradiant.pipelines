.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Fri 16 May 11:48:13 2014 CEST

.. include:: links.rst
.. testsetup:: *

   from bob.gradiant.pipelines import Pipeline, Pca, LinearSvc
   pipeline = Pipeline([Pca(name='FeaturesPca', n_components=0.95),LinearSvc(name='DetectorSvc')])
   current_directory = os.path.realpath(os.curdir)
   temp_dir = tempfile.mkdtemp(prefix='bob_doctest_')
   os.chdir(temp_dir)

============
 User Guide
============

By importing this package, you can use several machine learning `sklearn-pipeline`_-based utilities to train and classify your data.

A :py:class:`bob.gradiant.pipelines.Pipeline` receives the features and then following the next steps:

1. :py:func:`bob.gradiant.pipelines.Pipeline.fit`: Prepare the input data and train it
2. :py:func:`bob.gradiant.pipelines.Pipeline.run`: Predict the scores and prepare the outputdata
3. :py:func:`bob.gradiant.pipelines.Pipeline.save`: Save a pipeline model.


Writing routines to train and classify your data is easy using implemented :py:class:`bob.gradiant.pipelines.Processors`. You can chose a combination of these and use them into a :py:class:`bob.gradiant.pipelines.Pipeline`.
Note that the order of the list matters, so take it into account while you are defining your experiment.


Practical example

Let's show how it works with an example:

First we should load some data:
.. doctest::

    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_breast_cancer()
    >>> features = data['data']
    >>> labels = data['target']
    >>> features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,random_state=42)
    >>> x_train = {'features': features_train,'labels': labels_train} #prepare the data to fit pipeline input format
    >>> x_test = {'features': features_test,'labels': labels_test}

Now, we have to define a :py:class:`bob.gradiant.pipelines.Pipeline`. In this case a PCA + LinearSVC:

.. doctest::

   >>> from bob.gradiant.pipelines import Pipeline, Pca, LinearSvc
   >>> pipeline = Pipeline([Pca(name='FeaturesPca', n_components=0.95),LinearSvc(name='DetectorSvc')])

Then we can train you :py:class:`bob.gradiant.pipelines.Pipeline` with train subset:

.. doctest::

   >>> y_train = pipeline.fit_run(x_train)
   >>> print ('AUC for training set: ', roc_auc_score(x_train['labels'], y_train['scores'])) #overfitted

Once the :py:class:`bob.gradiant.pipelines.Pipeline` has finished the training we can evaluate our classifier using test data:

.. doctest::

   >>> y_test = pipeline.run(x_test)
   >>> print ('AUC for test set: ', roc_auc_score(x_test['labels'], y_test['scores']))

We can also save learned models for future experiments:
.. doctest::

   >>> pipeline.save(current_directory)

Otherwise, for loading a pipeline from saved models.

.. doctest::

   >>> pipeline.load(current_directory)

.. testcleanup:: *

  import shutil
  os.chdir(current_directory)
  shutil.rmtree(temp_dir)