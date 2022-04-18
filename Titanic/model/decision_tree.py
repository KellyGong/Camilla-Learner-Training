from sklearn.tree import DecisionTreeClassifier
from .scikit_base_object import ScikitBaseObject


class Decision_tree_clf(DecisionTreeClassifier, ScikitBaseObject):
    def __init__(self, **kwargs):
        DecisionTreeClassifier.__init__(self, **kwargs)
        ScikitBaseObject.__init__(self)
        self.name = 'decision_tree'
        self.kwargs = kwargs
        self.model_name = "decision_tree"
        for parameter in self.kwargs:
            self.model_name += '_' + str(self.kwargs[parameter])

    @classmethod
    def hyper(cls):
        hyper_dict = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_features': ['auto', 'sqrt', 'log2'],
        }
        return hyper_dict
