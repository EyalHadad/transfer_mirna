from src.models.csv_handler import save_cross_org_table, save_transfer_table
from src.models.transfer.Base_transfer import BaseTransferObj
from src.models.transfer.xgboost_transfer import XgboostTransferObj
from src.models.models_handler import *
import copy


def run_transfer(model_type, trans_epochs, transfer_dict, tran_folder):
    xgb_transfer_dict, base_transfer_dict = {}, {}
    xgb_vanilla_dict, base_vanilla_dict = {}, {}
    for src_model_name, src_dataset_list in transfer_dict.items():
        xgb_transfer_dict[src_model_name] = {}
        xgb_vanilla_dict[src_model_name] = {}
        base_transfer_dict[src_model_name] = {}
        base_vanilla_dict[src_model_name] = {}
        base_obj = BaseTransferObj(src_model_name)
        xgb_obj = XgboostTransferObj(src_model_name)
        dest_dict = copy.deepcopy(transfer_dict)
        del dest_dict[src_model_name]
        for dst_model_name, dst_dataset_list in dest_dict.items():
            base_obj.load_dst_data(dst_model_name,dst_dataset_list)
            xgb_obj.load_dst_data(dst_model_name,dst_dataset_list)
            xgb_transfer_dict[src_model_name][dst_model_name] = {}
            xgb_vanilla_dict[src_model_name][dst_model_name] = {}
            base_transfer_dict[src_model_name][dst_model_name] = {}
            base_vanilla_dict[src_model_name][dst_model_name] = {}
            for t_size in TRANSFER_SIZE_LIST:
                v_score = base_obj.train_new_model(t_size, 'base')
                base_vanilla_dict[src_model_name][dst_model_name][t_size] = v_score
                score = base_obj.retrain_model(t_size,model_type,trans_epochs)
                base_transfer_dict[src_model_name][dst_model_name][t_size] = score
                v_score = xgb_obj.train_new_model(t_size, 'xgboost')
                xgb_vanilla_dict[src_model_name][dst_model_name][t_size] = v_score
                score = xgb_obj.retrain_model(t_size, model_type, trans_epochs)
                xgb_transfer_dict[src_model_name][dst_model_name][t_size] = score
    save_transfer_table(base_transfer_dict, 'base', trans_epochs,tran_folder)
    save_transfer_table(base_vanilla_dict, 'base', 'baseline',tran_folder)
    save_transfer_table(xgb_transfer_dict, 'xgboost', trans_epochs,tran_folder)
    save_transfer_table(xgb_vanilla_dict, 'xgboost', 'baseline',tran_folder)



if __name__ == '__main__':
    transfer_folder_name = "dataset_models_" + datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    # run_training('base', TRAIN_DICT_REG, models_folder_name, 'datasets')
    run_transfer('base',20,TRAIN_DICT_REG,transfer_folder_name)
    # run_transfer(model_type='xgboost', trans_epochs=20)
    # for i in range(20,120,20):
    #     run_transfer(model_type='base', trans_epochs=i)
    #     print(i)
i = 9
