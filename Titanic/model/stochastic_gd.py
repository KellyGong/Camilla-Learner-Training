from sklearn.linear_model import SGDClassifier
from .scikit_base_object import ScikitBaseObject


class SGD_clf(SGDClassifier, ScikitBaseObject):
    def __init__(self, **kwargs):
        SGDClassifier.__init__(self, **kwargs)
        ScikitBaseObject.__init__(self)
        self.name = "SGD"
        self.kwargs = kwargs
        self.model_name = "SGD"
        for parameter in self.kwargs:
            self.model_name += '_' + str(self.kwargs[parameter])
    
    @classmethod
    def hyper(cls):
        hyper_dict = {
            'max_iter': [1e5],
            'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'eta0': [0.1, 0.001, 0.0001]
        }
        return hyper_dict
