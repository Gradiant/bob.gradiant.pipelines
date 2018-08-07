#!/usr/bin/env python
# Gradiant's Biometrics Team <biometrics.support@gradiant.org>
# Copyright (C) 2017 Gradiant, Vigo, Spain

import os.path
from bob.gradiant.pipelines import LinearSvc, Pca, Pipeline, PipelineSaver
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def main():
    data = load_breast_cancer()
    features = data['data']
    labels = data['target']
    features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                                labels,
                                                                                test_size=0.2,
                                                                                random_state=42)

    x_train = {
        'features': features_train,
        'labels': labels_train
    }
    x_test = {
        'features': features_test,
        'labels': labels_test
    }

    save_path = '/tmp/pipeline_example'
    intermediate_features_path = os.path.join(save_path, 'features')
    pipeline_train = Pipeline('Pca_and_linear_svc_pipeline',[Pca(n_components=0.95), LinearSvc(C=1.0)])

    print ('Training pipeline description: ', pipeline_train)

    y_train = pipeline_train.fit_run(x_train)
    print('AUC for training set: {}'.format(str(roc_auc_score(y_train['labels'], y_train['scores']))))

    pipeline_train.save(save_path)
    pca = Pca()
    pca.load(save_path)
    svc = LinearSvc()
    svc.load(save_path)

    pipeline_test = Pipeline('Pca_and_linear_svc_pipeline', [pca,
                                                             PipelineSaver(intermediate_features_path,
                                                                           'PCA_test_features'),
                                                             svc])
    pipeline_test.load(save_path)

    print('Test pipeline description: ', pipeline_test)

    y_test = pipeline_test.run(x_test)
    print('AUC for training set: {}'.format(str(roc_auc_score(y_test['labels'], y_test['scores']))))


if __name__ == '__main__':
    main()
