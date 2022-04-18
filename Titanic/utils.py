from pathlib import Path
import numpy as np
from itertools import product
import csv


class Hyper_Iterator:
    def __init__(self, **kwargs):
        self.parameters = kwargs.keys()
        values = [kwargs[param] for param in self.parameters]
        self.hyper_parameter_list = list(product(*values))
        self.index = len(self.hyper_parameter_list)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        hyper_parameter = {}
        for parameter, value in zip(self.parameters, self.hyper_parameter_list[self.index]):
            hyper_parameter[parameter] = value
        return hyper_parameter


def make_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)


def aggregate_model_result(path, data_row: list, csv_head: list):
    p = Path(path)
    if p.exists():
        with open(p, 'a+') as f:
            csv_writer = csv.writer(f, delimiter='\t')
            csv_writer.writerow(data_row)
    
    else:
        make_dir(p.parent)
        with open(p, 'w') as f:
            csv_writer = csv.writer(f, delimiter='\t')
            csv_writer.writerow(csv_head)
            csv_writer.writerow(data_row)


def save_ndarray(path, a):
    np.save(path, a)

def load_ndarray(path):
    return np.load(path)
