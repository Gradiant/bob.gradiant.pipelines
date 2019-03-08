import numpy as np
from sklearn.mixture import GaussianMixture
from bob.gradiant.pipelines.classes.processor import Processor
from bob.gradiant.pipelines.classes.default_keys_correspondences import DEFAULT_KEYS_CORRESPONDENCES
from bob.gradiant.pipelines.classes.processor_output_type import ProcessorOutputType
import pickle
import copy


class GmmOneClass(Processor):
    def __init__(self,
                 name='gmm',
                 n_components=1,
                 max_iter=20,
                 m_class=0,
                 keys_correspondences=DEFAULT_KEYS_CORRESPONDENCES):
        super(GmmOneClass, self).__init__(name)
        self._max_iter = max_iter
        self._n_components = n_components
        self._m_class = m_class
        self._model = GaussianMixture(n_components=self._n_components, max_iter=self._max_iter)
        self.keys_correspondences = keys_correspondences

    def to_dict(self):
        output_dict = {
            'data': np.array(pickle.dumps(self._model)),
        }
        return output_dict

    def from_dict(self, input_dict):
        self._model = pickle.loads(input_dict['data'])

    def fit(self, x):
        labels_key = self.keys_correspondences["labels_key"]
        features_key = self.keys_correspondences["features_key"]

        labels = copy.deepcopy(x[labels_key])
        if self._m_class not in labels:
            raise Exception('Selected class ', self._m_class, ' is not in features labels: ', np.unique(labels))
        features = x[features_key][labels == self._m_class]
        self._model.fit(features)

    def run(self, x):
        features_key = self.keys_correspondences["features_key"]
        scores_key = self.keys_correspondences["scores_key"]
        output_type_key = self.keys_correspondences["output_type_key"]

        x[scores_key] = self._model.predict_proba(x[features_key])
        x[output_type_key] = ProcessorOutputType.LIKELIHOOD
        return x

    def __str__(self):
        description = {
            'type': 'GaussianMixtureModel processor',
            'name': self.name,
            'n_components': self._n_components,
            'class to model': self._m_class
        }
        return str(description)
