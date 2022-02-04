from src.data.data_class import DataOrg
from src.models.models_builder import get_models_dict
from src.models.models_handler import *
from src.models.param_class import TrainModelParam, EvalModelParam


def run_transfer(training_dict, src_org, dst_org, folder_path):
    t_parm = TrainModelParam(folder_path=folder_path, to_save=False, epochs=200)
    e_parm = EvalModelParam(folder_path=folder_path)
    t_parm.data_obj = DataOrg(dst_org).load(datasets=training_dict[dst_org], is_train="train")
    e_parm.data_obj = DataOrg(dst_org).load(datasets=training_dict[dst_org], is_train="test")
    t_parm.src_model_to_load = src_org
    baseline_models = get_models_dict(len(e_parm.data_obj))
    t_parm.part_train = 500
    baseline_models['base'].train_model(t_parm=t_parm)
    baseline_models['base'].evaluate_model(e_parm=e_parm)


def run_improve(src_org='human', dst_org='human'):
    dir_path = list_files(MODELS_OBJECTS_PATH)[-1]
    run_transfer(VS4_REG_DICT, src_org, dst_org, dir_path)


if __name__ == '__main__':
    run_improve()
