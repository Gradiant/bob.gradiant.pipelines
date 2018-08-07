import numpy as np
from sklearn.mixture import GaussianMixture
from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.processor_output_type import ProcessorOutputType
import pickle
import copy


class GmmOneClass(Processor):
    def __init__(self, name='gmm', n_components=1, max_iter=20, m_class=0):
        super(GmmOneClass, self).__init__(name)
        self._max_iter = max_iter
        self._n_components = n_components
        self._m_class = m_class
        self._model = GaussianMixture(n_components=self._n_components, max_iter=self._max_iter)

    def to_dict(self):
        output_dict = {
            'data': np.array(pickle.dumps(self._model)),
        }

        return output_dict

    def from_dict(self, input_dict):
        self._model = pickle.loads(input_dict['data'])

    def fit(self, X):
        labels = copy.deepcopy(X['labels'])
        if self._m_class not in labels:
            raise Exception('Selected class ', self._m_class, ' is not in features labels: ', np.unique(labels))
        features = X['features'][labels == self._m_class]
        self._model.fit(features)

    def run(self, X):
        X['scores'] = self._model.predict_proba(X['features'])
        X['output_type'] = ProcessorOutputType.LIKELIHOOD
        return X

    def __str__(self):
        description = {
            'type': 'GaussianMixtureModel processor',
            'name': self.name,
            'n_components': self._n_components,
            'class to model': self._m_class
        }
        return str(description)