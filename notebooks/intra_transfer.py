import copy

from src.models.csv_handler import save_intra_transfer_table
from src.models.models_handler import *
from src.models.transfer.Base_transfer import BaseTransferObj
from src.models.transfer.xgboost_transfer import XgboostTransferObj


def run_intra_transfer(model_type='base', trans_epochs=20):
    dataset_dict = {'human': ['human1', 'human3'],
                    'worm': ['worm1', 'worm2'], 'mouse': ['mouse1', 'mouse2']}
    transfer_dict = {}
    vanilla_model_dict = {}
    for org, org_data_sets in dataset_dict.items():
        transfer_size = TRANSFER_SIZE_LIST
        for org_name in org_data_sets:
            rest = copy.deepcopy(org_data_sets)
            rest.remove(org_name)
            if model_type == 'base':
                trans_obj = BaseTransferObj(org_name)
            else:
                trans_obj = XgboostTransferObj(org_name)
            transfer_dict[org_name] = {}
            vanilla_model_dict[org_name] = {}
            for dst_org_name in rest:
                trans_obj.load_dst_data(dst_org_name)
                transfer_dict[org_name][dst_org_name] = {}
                vanilla_model_dict[org_name][dst_org_name] = {}
                for t_size in transfer_size:
                    v_auc = trans_obj.train_new_model(t_size, model_type)
                    vanilla_model_dict[org_name][dst_org_name][t_size] = v_auc
                    auc = trans_obj.retrain_model(t_size, model_type, trans_epochs)
                    transfer_dict[org_name][dst_org_name][t_size] = auc
    save_intra_transfer_table(transfer_dict, model_type, trans_epochs)
    save_intra_transfer_table(vanilla_model_dict, model_type, 'baseline')


if __name__ == '__main__':
    run_intra_transfer(model_type='base', trans_epochs=20)
    run_intra_transfer(model_type='xgboost', trans_epochs=20)
