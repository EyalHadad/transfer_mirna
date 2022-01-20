import copy
import csv
import os
from datetime import datetime

import pandas as pd

from constants import *
from constants import SPECIES, TRANSFER_SIZE_LIST


def save_intra_dataset(table_dict, model_type):
    data = pd.DataFrame.from_dict(table_dict, orient='index')
    f_name = os.path.join(MODELS_INTRA_TABELS, f"{model_type}.csv")
    data.to_csv(f_name)


def load_intra_tabel(model_type, tabel_dict, aa1=""):
    df = pd.read_csv(os.path.join(MODELS_INTRA_TABELS, f"{aa1}{model_type}.csv"))
    in_res_dict = dict(zip(df[df.columns[0]], df[df.columns[1]]))
    res = pd.DataFrame(index=tabel_dict.keys(), columns=tabel_dict.keys())
    return in_res_dict, res


def save_cross_org_table(tabel_dict, model_type):
    print("--- Saving tabel results ---")
    in_res_dict, res = load_intra_tabel(model_type, tabel_dict, aa1)
    for r in tabel_dict.keys():
        res.at[r, r] = round(in_res_dict[r], 2)
        for c in tabel_dict[r].keys():
            res.at[r, c] = round(tabel_dict[r][c][0], 2)
    time_var = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    res.to_csv(os.path.join(MODELS_CROSS_ORG_TABELS, f"{model_type}_{time_var}.csv"))


def save_intra_transfer_table(transfer_dict, model_type, trans_epochs):
    f_name = os.path.join(MODELS_INTRA_TRANSFER_TABLES, f"{model_type}_{trans_epochs}_transfer.csv")
    with open(f_name, 'w') as f:
        f_header = ['src_org', 'dst_org'] + TRANSFER_SIZE_LIST + ['\n']
        f.write(','.join(str(x) for x in f_header))
        for s_org, v in transfer_dict.items():
            for d_org, t_values in transfer_dict[s_org].items():
                t_values = []
                for t_size, value in transfer_dict[s_org][d_org].items():
                    t_values.append(str(round(value, 2)))
                row_to_write = [s_org, d_org] + [val for val in t_values] + ['\n']
                f.write(','.join(row_to_write))


def save_aa1_transfer_table(table_dict, model_type, trans_epochs):
    transfer_dict = {}

    for src_org in table_dict.keys():
        if src_org not in transfer_dict:
            transfer_dict[src_org] = {}
        for dst_org in table_dict[src_org].keys():
            if dst_org not in transfer_dict[src_org]:
                transfer_dict[src_org][dst_org] = {}
            for s, val in table_dict[src_org][dst_org].items():
                if s not in transfer_dict[src_org][dst_org]:
                    transfer_dict[src_org][dst_org][s] = []
                transfer_dict[src_org][dst_org][s].append(val)
    f_name = os.path.join(MODELS_OBJECTS_TRANSFER_TABLES, f"aa1_{model_type}_{trans_epochs}_transfer.csv")
    with open(f_name, 'w') as f:
        f_header = ['src_org', 'dst_org'] + TRANSFER_SIZE_LIST + ['\n']
        f.write(','.join(str(x) for x in f_header))
        for s_org, v in transfer_dict.items():
            for d_org, t_values in transfer_dict[s_org].items():
                t_values = []
                for t_size, values in transfer_dict[s_org][d_org].items():
                    l_str = [str(round(x, 2)) for x in values]
                    t_values.append(":".join(l_str))

                row_to_write = [s_org, d_org] + [val for val in t_values] + ['\n']
                f.write(','.join(row_to_write))


def save_transfer_table(table_dict, model_type, trans_epochs, tran_folder):
    transfer_dict = create_empty_species_dict()

    for src_org in table_dict.keys():
        for dst_org in table_dict.keys():
            if dst_org[:-1] in transfer_dict[src_org[:-1]]:
                for s, val in table_dict[src_org][dst_org].items():
                    transfer_dict[src_org[:-1]][dst_org[:-1]][s].append(val)
    f_name = os.path.join(MODELS_OBJECTS_TRANSFER_TABLES, f"{tran_folder}/{model_type}_{trans_epochs}_transfer.csv")
    with open(f_name, 'w') as f:
        f_header = ['src_org', 'dst_org'] + TRANSFER_SIZE_LIST + ['\n']
        f.write(','.join(str(x) for x in f_header))
        for s_org, v in transfer_dict.items():
            for d_org, t_values in transfer_dict[s_org].items():
                t_values = []
                for t_size, values in transfer_dict[s_org][d_org].items():
                    l_str = [str(round(x, 2)) for x in values]
                    t_values.append(":".join(l_str))

                row_to_write = [s_org, d_org] + [val for val in t_values] + ['\n']
                f.write(','.join(row_to_write))


def save_metrics(eval_dict):
    f_path = MODELS_PATH / 'models_evaluation.csv'
    with open(f_path, 'a') as file:
        writer = csv.DictWriter(file, eval_dict.keys(), delimiter=',', lineterminator='\n')
        if file.tell() == 0:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow(eval_dict)


def save_feature_importance_res(row_desc, f_importance_list, type_name):
    f_path = os.path.join(MODELS_FEATURE_IMPORTANCE, 'models_feature_importance_{0}.csv'.format(type_name))
    f_importance_str = ",".join(["{0}:{1}".format(x[0], x[1]) for x in f_importance_list[:10]])
    date_time = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    with open(f_path, 'a') as file:
        file.write('{0},{1},{2}'.format(row_desc, date_time, f_importance_str))
        file.write("\n")


def create_empty_species_dict():
    di = {}
    for a in SPECIES:
        di[a] = dict()
    for a in SPECIES:
        rest = copy.deepcopy(SPECIES)
        rest.remove(a)
        for r in rest:
            di[a][r] = {}
            for s in TRANSFER_SIZE_LIST:
                di[a][r][s] = []
    return di
