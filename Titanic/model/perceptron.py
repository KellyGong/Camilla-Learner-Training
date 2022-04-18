from sklearn.linear_model import Perceptron
from .scikit_base_object import ScikitBaseObject


class Perceptron_clf(Perceptron, ScikitBaseObject):
    def __init__(self, **kwargs):
        Perceptron.__init__(self, **kwargs)
        ScikitBaseObject.__init__(self)
        self.name = "perceptron"
        self.kwargs = kwargs
        self.model_name = "perceptron"
        for parameter in self.kwargs:
            self.model_name += '_' + str(self.kwargs[parameter])
    
    @classmethod
    def hyper(cls):
        hyper_dict = {
            'penalty': ['l2', 'l1', 'elasticnet'],
            'eta0': [1.0, 0.1, 0.5],
            'early_stopping': [True, False]
        }
        return hyper_dict
