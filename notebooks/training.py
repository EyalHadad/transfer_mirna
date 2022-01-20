from datetime import datetime

from src.data.data_class import DataOrg, ScoreObj
from src.data.data_handler import *
from src.models.models import get_models_dict


def run_training(training_dict, model_list, metrics, folder_name):
    scores = ScoreObj(model_list, metrics)
    for src_org_name, datasets in training_dict.items():
        train_data = DataOrg(src_org_name).load(datasets=datasets, is_train="train", val_ratio=0.1)
        test_data = DataOrg(src_org_name).load(datasets=datasets, is_train="test")
        models_dict = get_models_dict(len(train_data))
        for model_name in model_list:
            model = models_dict[model_name]
            model.train_model(data_obj=train_data, folder_path=MODELS_OBJECTS_PATH / folder_name)
            score_dict = model.evaluate_model(model_name=src_org_name, test_data=test_data,
                                              models_folder=MODELS_OBJECTS_PATH / folder_name)
            scores.add_score(model_name, score_dict, src_org_name)

    scores.save_results(MODELS_OBJECTS_PATH / folder_name)


if __name__ == '__main__':
    f_time_name = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    os.mkdir(MODELS_OBJECTS_PATH / f_time_name)
    run_training(VS4_REG_DICT, ['base', 'xgb'], ['ACC', 'F1_score'], f_time_name)
