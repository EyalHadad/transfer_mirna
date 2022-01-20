import os
import pickle
from datetime import datetime

import numpy as np
from sklearn import metrics

from constants import *
from src.models.csv_handler import save_metrics


def save_pkl_model(model, pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def create_evaluation_dict(t_model_name, org_name, pred, y):
    model_name = '{0}_{1}'.format(t_model_name, org_name)
    date_time = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    print(f" There are {np.sum(np.isnan(pred))} nan predictions")
    np.nan_to_num(pred, copy=False)
    print(f" After filling 0 instead of nan there are {np.sum(np.isnan(pred))} nan predictions")
    eval_dict = {'Model': model_name, 'Date': date_time, 'ACC': metrics.accuracy_score(y, np.round(pred))}
    eval_dict['FPR'], eval_dict['TPR'], thresholds = metrics.roc_curve(y, pred)
    eval_dict['AUC'] = metrics.auc(eval_dict['FPR'], eval_dict['TPR'])
    eval_dict['PR'] = metrics.precision_score(y, np.round(pred), average='micro')
    eval_dict['F1_score'] = metrics.f1_score(y, np.round(pred))
    save_metrics(eval_dict)
    print("ACC:", eval_dict['ACC'], "___PR:", eval_dict['PR'])
    return eval_dict


def create_res_graph(tabel_dict, org_name, model_type, trans_epochs):
    f_header = ['model'] + TRANSFER_SIZE_LIST + ['\n']
    for c in tabel_dict.keys():  # assume that c is creature
        f_name = os.path.join(MODELS_OBJECTS_GRAPHS, f"{org_name}_{c}.csv")
        with open(f_name, 'a') as file:
            if file.tell() == 0:
                file.write(','.join(str(x) for x in f_header))
            row_to_write = [f"{model_type}_{trans_epochs}"] + [str(round(tabel_dict[c][t], 2)) for t in
                                                               tabel_dict[c].keys()] + ['\n']
            file.write(','.join(row_to_write))
