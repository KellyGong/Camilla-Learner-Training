from model import *
from preprocess import prepare_dataset
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from utils import *
from tqdm import tqdm
import random


clf_models = [Log_clf, Decision_tree_clf, Svc_clf, LinearSvc_clf, Knn_clf,
              RandomForest_clf, Gaussian_clf, Perceptron_clf, SGD_clf]
x_train, y_train, x_test, y_test = prepare_dataset()


def gen(model, x_train, y_train, x_test, y_test):
    y_pred = cross_val_predict(model, x_train, y_train, cv=5)
    cv_score = accuracy_score(y_train, y_pred)
    y_correct = np.array(y_pred == np.array(y_train), dtype=np.int)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    test_correct = np.array(y_pred == np.array(y_test), dtype=np.int)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return model, y_correct, test_correct, {'cv_score': cv_score, 'accuracy_score': accuracy, 'f1_score': f1}


def save(model, x_train, y_train, x_test, y_test, y_correct, test_correct, result):
    aggregate_result_path = f'result/model_prediction/{model.name}/aggregate.csv'
    path = f'result/model_prediction/{model.name}/{model.model_name}'

    # save data file
    make_dir(path)
    save_ndarray(path + '/x_train.npy', x_train)
    save_ndarray(path + '/y_train.npy', y_train)
    save_ndarray(path + '/x_test.npy', x_test)
    save_ndarray(path + '/y_test.npy', y_test)
    save_ndarray(path + '/y_correct.npy', y_correct)
    save_ndarray(path + '/test_correct.npy', test_correct)

    # save model
    model.save_model(path + f'/{model.name}.model')
    model.save_results(path + '/result.json', result)
    model.save_hyper_parameters(path + '/hyper.json')

    # write each model result to parent model file
    data_row = [str(round(value, 4)) for value in result.values()]
    data_row.append(model.model_name)
    csv_head = list(result.keys())
    csv_head.append('model')
    aggregate_model_result(aggregate_result_path, data_row, csv_head)


def load_model(path):
    p = Path(path)
    model_files = list(p.glob('*.model'))
    assert len(model_files) == 1
    model = ScikitBaseObject.load_model(model_files[0])
    return model


def gen_models_result():
    for model_object in tqdm(clf_models):
        hyper_model = model_object.hyper()
        hyper_iterator = Hyper_Iterator(**hyper_model)
        for hyper_parameter in hyper_iterator:
            model = model_object(**hyper_parameter)
            try:
                model, y_correct, test_correct, result = gen(model, x_train, y_train, x_test, y_test)
                if not model.check_exist_result(tuple(list(result.values()))):
                    save(model, x_train, y_train, x_test, y_test, y_correct, test_correct, result)
            except Exception:
                pass


def load_response_matrix(path='result/model_prediction', model_num=-1):
    p = Path(path)
    respones_files = list(p.glob('**/y_correct.npy'))
    model_name_list = []
    response_matrix = []
    # FIXME
    random.shuffle(respones_files)
    len_response_files = len(respones_files)
    if model_num > 0:
        respones_files = respones_files[0: model_num]
    for response_file in respones_files:
        model_name = response_file.parts[-2]
        model_name_list.append(model_name)
        response_matrix.append(load_ndarray(response_file))
    response_matrix = np.vstack(response_matrix)

    # response_tuple
    rows, cols = np.where(response_matrix >= 0)
    responses = response_matrix[rows, cols]
    response_tuple = [(row, col, response) for (row, col, response) in zip(rows, cols, responses)]
    # print(f'drop_rate: {drop_learners_rate}')
    print(f'model_num: {model_num}')
    return response_matrix, response_tuple, model_name_list


if __name__ == '__main__':
    # for model_object in tqdm(clf_models):
    #     model = model_object()
    #     model, y_correct, result = gen(model, x_train, y_train, x_test, y_test)
    #     save(model, x_train, y_train, x_test, y_test, y_correct, result)
    # load_response_matrix()
    # load_model('result/model_prediction/knn')
    # model_hyper = Knn_clf.hyper()
    # hyper_iterator = Hyper_Iterator(**model_hyper)
    # for hyper_parameter in hyper_iterator:
    #     model = Knn_clf(**hyper_parameter)
    gen_models_result()
    # load_response_matrix()
    print('yes')
