from sklearn.linear_model import LogisticRegression
from .scikit_base_object import ScikitBaseObject


class Log_clf(LogisticRegression, ScikitBaseObject):
    def __init__(self, max_iter = 10000, random_state = 7, **kwargs):
        LogisticRegression.__init__(self, max_iter=max_iter, random_state=random_state, **kwargs)
        ScikitBaseObject.__init__(self)
        self.name = "logistic_regression"
        self.kwargs = kwargs
        self.model_name = "logistic_regression"
        for parameter in self.kwargs:
            self.model_name += '_' + str(self.kwargs[parameter])
    
    @classmethod
    def hyper(cls):
        hyper_dict = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.1, 0.5, 1, 5, 10],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'warm_start': [True, False]
        }
        return hyper_dict
