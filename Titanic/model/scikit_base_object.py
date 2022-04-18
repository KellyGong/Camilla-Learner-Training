from joblib import dump, load
import json


class ScikitBaseObject:
    def __init__(self):
        self.trained_parameters = set()

    def save_model(self, path):
        dump(self, path)
    
    @classmethod
    def load_model(cls, path):
        return load(path)
    
    @classmethod
    def save_results(cls, path, result_dict):
        with open(path, 'w', encoding='utf8') as f:
            json.dump(result_dict, f, indent=4)
    
    def save_hyper_parameters(self, path):
        with open(path, 'w', encoding='utf8') as f:
            json.dump(self.kwargs, f, indent=4)

    def check_exist_result(self, result_values: tuple):
        exist = False
        if result_values in self.trained_parameters:
            exist = True
        else:
            self.trained_parameters.add(result_values)
        return exist
