from sklearn.naive_bayes import GaussianNB
import os
from constants import *
from time import gmtime, strftime
from src.models.model_learner import ModelLearner
from src.models.models_handler import save_pkl_model



class NbTrainObj(ModelLearner):

    def __init__(self,org_name,m_name):
        ModelLearner.__init__(self,org_name,m_name)


    def train_model(self):
        super().prep_model_training()
        print("---Start training {0} on {1}---\n".format(self.model_name, self.org_name))
        self.model = GaussianNB()
        self.model.fit(self.x, self.y)
        model_name = os.path.join(MODELS_OBJECTS_PATH,
                                  'NB_{0}_{1}.pkl'.format(self.org_name, strftime("%Y-%m-%d", gmtime())))
        save_pkl_model(self.model, model_name)
        print("---{0} model saved---\n".format(self.model_name))



    def model_explain(self):
        print("---Explain model---\n")
        super().calc_permutation_importance()








