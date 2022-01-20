import logging
import os

import matplotlib.pyplot as plt
from keras.optimizers import SGD

from constants import *
from src.data.data_class import DataOrg
from src.models.model_learner import ModelLearner, create_evaluation_dict

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)


class BaseTrainObj(ModelLearner):
    history = None

    def __init__(self):
        self.model_name = 'base'
        ModelLearner.__init__(self)

    def set_model(self, model):
        self.model = model
        self.model.compile(optimizer=SGD(lr=0.01, momentum=0.9, clipnorm=1.0), loss='binary_crossentropy',
                           metrics=['acc'])

    def train_model(self, data_obj: DataOrg, folder_path):
        self.history = self.model.fit(data_obj.data, data_obj.label, epochs=1,
                                      validation_data=(data_obj.v_data, data_obj.v_label))
        self.plot_learning_curves(folder_path, data_obj.dataset_name)
        model_folder_name = folder_path / f"{data_obj.dataset_name}/"
        if not os.path.isdir(model_folder_name):
            os.mkdir(model_folder_name)
        self.model.save_weights(str(folder_path / f"{data_obj.dataset_name}") + "/")

    def plot_learning_curves(self, folder_name, org_name):
        f_name = folder_name / "learning_results"
        if not os.path.isdir(f_name):
            os.mkdir(f_name)

        for metric in ['acc', 'loss']:
            self.draw_curve(metric, f_name, org_name)

    def draw_curve(self, metric, f_name, org_name):
        reg_metric = self.history.history[metric]
        val_metric = self.history.history[f"val_{metric}"]
        epochs = range(1, len(reg_metric) + 1)
        plt.figure()
        plt.plot(epochs, reg_metric, 'bo', label=f"Training {metric}_metric")
        plt.plot(epochs, val_metric, 'b', label=f"Validation {metric}_metric")
        plt.title(f"{metric} {self.model_name}")
        plt.legend()
        plt.savefig(f_name / f"{self.model_name}_{metric}_{org_name}.png")
        plt.clf()

    def evaluate_model(self, model_name, test_data: DataOrg, models_folder: Path, to_load=False):
        if to_load:
            self.model.load_weights(models_folder / model_name)
        pred = self.model.predict(test_data.data)
        eval_dict = create_evaluation_dict(self.model_name, model_name, pred, test_data.label)
        return eval_dict
