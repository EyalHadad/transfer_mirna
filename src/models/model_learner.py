from abc import abstractmethod

import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance

from src.data.data_class import DataOrg
from src.models.models_handler import *


class ModelLearner:
    model_name = None
    model = None

    def __init__(self):
        pass

    @abstractmethod
    def set_model(self, input_shape):
        raise NotImplementedError("Must override get_shap_values method")

    @abstractmethod
    def train_model(self, data_obj: DataOrg, folder_path):
        raise NotImplementedError("Must override train_model method")

    def plot_learning_curves(self, folder_name, org_name):
        pass

    @abstractmethod
    def evaluate_model(self, model_name, test_data: DataOrg, models_folder: Path, to_load=False):
        raise NotImplementedError("Must override evaluate_model method")

    def model_explain(self):
        print("Shap values\n")
        if self.model_name == 'Base':
            explainer = shap.DeepExplainer(self.model, self.x.values.astype('float'))
        else:
            explainer = shap.TreeExplainer(self.model, self.x.values.astype('float'))
        shap_values = explainer.shap_values(self.xval.values)
        shap.summary_plot(shap_values, feature_names=self.xval.columns, show=False,
                          max_display=10, plot_size=(20, 20))
        plt.title('{0} {1} SHAP bar'.format(self.model_name, self.org_name))
        plt.savefig(os.path.join(MODELS_FEATURE_IMPORTANCE, '{0}_{1}_bar.png'.format(self.model_name, self.org_name)))
        plt.clf()
        print("---Shap plots were saved---\n")

    def calc_permutation_importance(self):
        imps = permutation_importance(self.model, self.xval, self.yval)
        importances = imps.importances_mean
        std = imps.importances_std
        indices = np.argsort(importances)[::-1]
        title = '{0} {1} f_important'.format(self.model_name, self.org_name)
        # TODO move it to handler file (plot_figure)- need to know first what argument to send
        plt.figure(figsize=(10, 7))
        plt.title(title)
        plt.bar(range(self.xval.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
        # plt.xticks(range(X_test.shape[1]), [features[indices[i]] for i in range(6)])
        # plt.xlim([-1, X_test.shape[1]])
        plt.savefig(os.path.join(MODELS_FEATURE_IMPORTANCE, '{0}_bar.png'.format(title)))
        plt.clf()

    def plot_roc_curve(self, model_type, pred, y):
        from sklearn.metrics import roc_curve
        fpr1, tpr1, thresh1 = roc_curve(y, pred, pos_label=1)
        random_probs = [0 for i in range(len(y))]
        p_fpr, p_tpr, _ = roc_curve(y, random_probs, pos_label=1)
        plt.style.use('seaborn')
        plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='Model')
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        # skplt.metrics.plot_roc_curve(y.to_numpy().reshape(y.shape[0],1), pred)
        plt.savefig(os.path.join(MODELS_OUTPUT_PATH, f"ROC_{model_type}_{self.org_name}.png"))
        plt.clf()


@abstractmethod
def get_shap_values():
    raise NotImplementedError("Must override get_shap_values method")
