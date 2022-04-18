from sklearn.ensemble import RandomForestClassifier
from .scikit_base_object import ScikitBaseObject


class RandomForest_clf(RandomForestClassifier, ScikitBaseObject):
    def __init__(self, **kwargs):
        RandomForestClassifier.__init__(self, **kwargs)
        ScikitBaseObject.__init__(self)
        self.name = "random_forest"
        self.kwargs = kwargs
        self.model_name = "random_forest"
        for parameter in self.kwargs:
            self.model_name += '_' + str(self.kwargs[parameter])
    
    @classmethod
    def hyper(cls):
        hyper_dict = {
            'n_estimators': [10, 50, 100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, None],
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_samples_leaf': [1, 3, 5]
        }
        return hyper_dict
