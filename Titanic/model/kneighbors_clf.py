from sklearn.neighbors import KNeighborsClassifier
from .scikit_base_object import ScikitBaseObject


class Knn_clf(KNeighborsClassifier, ScikitBaseObject):
    def __init__(self, **kwargs):
        KNeighborsClassifier.__init__(self, **kwargs)
        ScikitBaseObject.__init__(self)
        self.name = "knn"
        self.kwargs = kwargs
        self.model_name = "knn"
        for parameter in self.kwargs:
            self.model_name += '_' + str(self.kwargs[parameter])

    @classmethod
    def hyper(cls):
        hyper_dict = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        }
        return hyper_dict
