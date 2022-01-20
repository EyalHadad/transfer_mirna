from src.data.data_class import ScoreObj, DataOrg
from src.models.csv_handler import *
from src.models.models import get_models_dict


def create_cross_org_tables(training_dict, model_list, metrics, folder_name):
    scores = ScoreObj(model_list, metrics)
    for src_org_name, src_dataset_list in training_dict.items():
        for dst_org_name, dst_dataset_list in training_dict.items():
            test_data = DataOrg(dst_org_name).load(datasets=dst_dataset_list, is_train="test")
            models_dict = get_models_dict(len(test_data))
            for model_name in model_list:
                model = models_dict[model_name]
                score_dict = model.evaluate_model(model_name=src_org_name, test_data=test_data,
                                                  models_folder=MODELS_OBJECTS_PATH / folder_name, to_load=True)
                scores.add_score(model_name=model_name, score_dict=score_dict, org_name=dst_org_name)
    scores.save_results(MODELS_PATH / "cross_org_tabels" / datetime.now().strftime("%d_%m_%Y %H_%M_%S"))


if __name__ == '__main__':
    f_time_name = "asd"
    create_cross_org_tables(VS4_REG_DICT, ['base,xgb'], ['ACC', 'F1_score'], f_time_name)
