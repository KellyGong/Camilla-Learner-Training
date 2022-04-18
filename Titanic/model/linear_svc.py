from sklearn.svm import SVC, LinearSVC
from .scikit_base_object import ScikitBaseObject


class Svc_clf(SVC, ScikitBaseObject):
    def __init__(self, max_iter = 1000, random_state = 7, **kwargs):
        SVC.__init__(self, max_iter=max_iter, random_state=random_state, **kwargs)
        ScikitBaseObject.__init__(self)
        self.name = "SVC"
        self.kwargs = kwargs
        self.model_name = "SVC"
        for parameter in self.kwargs:
            self.model_name += '_' + str(self.kwargs[parameter])
    
    @classmethod
    def hyper(cls):
        hyper_dict = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            'gamma': ['scale', 'auto'],
            'coef0': [.0, .1, 1.0],
            'tol': [1e-3, 1e-4, 1e-2]
        }
        return hyper_dict


class LinearSvc_clf(LinearSVC, ScikitBaseObject):
    def __init__(self, max_iter = 100000, random_state = 7, **kwargs):
        LinearSVC.__init__(self, max_iter=max_iter, random_state=random_state, **kwargs)
        ScikitBaseObject.__init__(self)
        self.name = "Linear_SVC"
        self.kwargs = kwargs
        self.model_name = "Linear_SVC"
        for parameter in self.kwargs:
            self.model_name += '_' + str(self.kwargs[parameter])
    
    @classmethod
    def hyper(cls):
        hyper_dict = {
            'penalty': ['l1', 'l2'],
            'loss': ['hinge', 'squared_hinge'],
            'C': [.5, 2.0, 1.0],
            'tol': [1e-3, 1e-4, 1e-2]
        }
        return hyper_dict
