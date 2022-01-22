import csv
import os
import pickle
from datetime import datetime
from typing import List

import numpy as np
from sklearn import metrics

from constants import *


def save_pkl_model(model, pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def create_evaluation_dict(t_model_name, org_name, pred, y):
    model_name = '{0}_{1}'.format(t_model_name, org_name)
    date_time = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    np.nan_to_num(pred, copy=False)
    eval_dict = {'Model': model_name, 'Date': date_time, 'ACC': metrics.accuracy_score(y, np.round(pred))}
    eval_dict['FPR'], eval_dict['TPR'], thresholds = metrics.roc_curve(y, pred)
    eval_dict['AUC'] = metrics.auc(eval_dict['FPR'], eval_dict['TPR'])
    eval_dict['PR'] = metrics.precision_score(y, np.round(pred), average='micro')
    eval_dict['F1_score'] = metrics.f1_score(y, np.round(pred))
    save_metrics(eval_dict)
    return eval_dict


def save_metrics(eval_dict):
    f_path = MODELS_PATH / 'models_evaluation.csv'
    with open(f_path, 'a') as file:
        writer = csv.DictWriter(file, eval_dict.keys(), delimiter=',', lineterminator='\n')
        if file.tell() == 0:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow(eval_dict)


def create_dir_if_not_exists(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def create_dir(parent_path: Path):
    f_time_name = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    dir_path = parent_path / f_time_name
    create_dir_if_not_exists(dir_path)
    return dir_path


def list_files(path: Path) -> List[Path]:
    dir_entries = sorted(os.scandir(path),
                         key=lambda file_entry: Path(file_entry).stem)
    return [Path(dir_entry.path) for dir_entry in dir_entries]
