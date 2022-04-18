from sklearn.naive_bayes import GaussianNB
from .scikit_base_object import ScikitBaseObject


class Gaussian_clf(GaussianNB, ScikitBaseObject):
    def __init__(self, **kwargs):
        GaussianNB.__init__(self, **kwargs)
        ScikitBaseObject.__init__(self)
        self.name = "gaussianNB"
        self.kwargs = kwargs
        self.model_name = "gaussianNB"
        for parameter in self.kwargs:
            self.model_name += '_' + str(self.kwargs[parameter])
    
    @classmethod
    def hyper(cls):
        hyper_dict = {
            'var_smoothing': [1e-9, 1e-8]
        }
        return hyper_dict
    

