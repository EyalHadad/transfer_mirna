import os

import matplotlib.pyplot as plt

from constants import *
from src.data.data_class import DataOrg
from src.models.csv_handler import save_feature_importance_res
from src.models.model_learner import ModelLearner
from src.models.models_handler import create_evaluation_dict


class XgboostTrainObj(ModelLearner):

    def __init__(self):
        self.model_name = 'xgb'
        ModelLearner.__init__(self)

    def set_model(self, model):
        self.model = model

    def train_model(self, data_obj: DataOrg, folder_path):
        self.model = self.model.fit(data_obj.data, data_obj.label, eval_metric=["error", "logloss"],
                                    eval_set=[(data_obj.v_data, data_obj.v_label)])

        # self.plot_learning_curves(folder_path,data_obj.dataset_name)  # TODO check where self.model.evals_result() val_1
        self.model.save_model(folder_path / f"{data_obj.dataset_name}.dat")

    def plot_learning_curves(self, folder_name, org_name):
        f_name = folder_name / "learning_results"
        if not os.path.isdir(f_name):
            os.mkdir(f_name)

        for metric in ['logloss', 'error']:
            self.draw_curve(metric, f_name, org_name)

    def draw_curve(self, metric, f_name, org_name):
        results = self.model.evals_result()
        epochs = len(results['validation_0'][metric])
        x_axis = range(0, epochs)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(x_axis, results['validation_0'][metric], label='Train')
        ax.plot(x_axis, results['validation_1'][metric], label='Test')
        ax.legend()
        plt.ylabel(metric)
        plt.title(f"XGBoost {metric}")
        plt.savefig(f_name / f"{self.model_name}_{metric}_{org_name}.png")
        plt.clf()

    def evaluate_model(self, model_name, test_data: DataOrg, models_folder: Path, to_load=False):
        if to_load:
            self.model.load_model(models_folder / f"{model_name}.dat")
        pred = self.model.predict_proba(test_data.data)[:, 1]
        eval_dict = create_evaluation_dict(self.model_name, model_name, pred, test_data.label)
        return eval_dict

    def model_explain(self):
        print("---Explain model---\n")
        # self.feature_importance()
        super().model_explain()

    def feature_importance(self):
        print("feature_importances\n")
        importance = self.model.feature_importances_
        f_important = sorted(list(zip(self.feature_names, importance)), key=lambda x: x[1], reverse=True)
        save_feature_importance_res('{0}_{1}'.format(self.model_name, self.org_name), f_important, 'reg')

        plt.bar([x[0] for x in f_important[:5]], [x[1] for x in f_important[:5]])
        plt.xticks(rotation=20)
        title = '{0} {1} f_important'.format(self.model_name, self.org_name)
        plt.title(title)
        plt.savefig(os.path.join(MODELS_FEATURE_IMPORTANCE, '{0}.png'.format(title)))
        plt.clf()
