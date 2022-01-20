from src.models.models_handler import create_sequence, load_trained_model,create_new_model
from src.models.csv_handler import save_metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from constants import *
from time import gmtime, strftime
from src.models.model_learner import ModelLearner
from src.models.miRNA_transfer_subclass import miTransfer, api_model
import numpy as np
import pandas as pd
import logging
from keras.layers import Input

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
import tensorflow as tf
from tensorflow.python.keras.models import save_model
from tensorflow.python.keras.models import load_model
from src.models.models_handler import *
from keras.optimizers import SGD
import xgboost as xgb




class Transfer_obj:
    feature_names = None
    src_model_name = None
    dst_org_name = None
    transfer_size = None
    l_model = None
    l_model_path = None
    x = None
    y = None
    x_train = None
    y_train = None
    sequences = None
    x_test = None
    y_test = None
    sequences_tst = None

    def __init__(self, org_name):
        self.src_model_name = org_name
        self.src_org_name = org_name

    def load_dst_data(self, dst_org_name,dataset_list):
        self.dst_org_name = dst_org_name
        train = datasets_list_data_extraction(dataset_list, "train")
        test = datasets_list_data_extraction(dataset_list, "test")

        print("---Data was loaded---\n")
        print("Train data shape:", train.shape)
        print("Test data shape:", test.shape)
        train['sequence'] = train.apply(lambda x: create_sequence(x['miRNA sequence'], x['target sequence']), axis=1)
        test['sequence'] = test.apply(lambda x: create_sequence(x['miRNA sequence'], x['target sequence']), axis=1)
        self.y = train['label']
        self.y_test = test['label']
        self.sequences = np.array(train['sequence'].values.tolist())
        self.sequences_tst = np.array(test['sequence'].values.tolist())
        X = train.drop(FEATURES_TO_DROP, axis=1)
        self.feature_names = list(X.columns)
        self.x = X.drop('sequence', 1).fillna(0)
        self.x = self.x.astype("float")
        X_test = test.drop(FEATURES_TO_DROP, axis=1)
        self.x_test = X_test.drop('sequence', 1).fillna(0)
        self.x_test = self.x_test.astype("float")

    def retrain_model(self, t_size, obj_type, trans_epochs,aa1=False):
        if t_size == 0:
            if obj_type == 'xgboost':
                self.l_model = xgb.XGBClassifier(kwargs=XGBS_PARAMS)  # init model
                self.l_model.load_model(self.l_model_path)

        else:
            x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(self.x, self.y, train_size=t_size,random_state=42)
            if obj_type == 'base':
                self.l_model.fit(x_train_t, y_train_t, epochs=trans_epochs)
            elif obj_type == 'xgboost':
                self.l_model = xgb.XGBClassifier(kwargs=XGBS_PARAMS)  # init model
                self.l_model.fit(x_train_t, y_train_t, early_stopping_rounds=trans_epochs,xgb_model=self.l_model_path,
                                 eval_set=[(x_test_t.iloc[:5, ], y_test_t.iloc[:5, ])])

        auc = self.eval_model(t_size, obj_type)
        return auc

    def train_new_model(self, t_size, obj_type):
        if t_size == 0:
            return 0

        x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(self.x, self.y, train_size=t_size, random_state=42)
        model = create_new_model(obj_type)
        if obj_type == 'base':
            model.fit(x_train_t, y_train_t, epochs=20)
        elif obj_type == 'xgboost':
            model.fit(x_train_t, y_train_t, early_stopping_rounds=20,
                      eval_set=[(x_test_t.iloc[:5, ], y_test_t.iloc[:5, ])])
        pred = model.predict(self.x_test)
        m_name = f"{obj_type}_{self.src_model_name}_{str(t_size)}"
        date_time, org_name, auc = create_evaluation_dict(m_name, 'baseline', pred, self.y_test)
        return auc

    def eval_model(self, t_size, obj_type):
        print("Evaluate model")
        if obj_type=='base':
            pred = self.l_model.predict(self.x_test)
        else:
            pred = self.l_model.predict_proba(self.x_test)[:,1]
        m_name = f"{obj_type}_{self.src_model_name}_{str(t_size)}"
        date_time, org_name, auc = create_evaluation_dict(m_name, self.dst_org_name, pred, self.y_test)
        if t_size==0:
            total_frame = self.x_test.copy()
            total_frame["index"] = self.x_test.index
            total_frame["actual"] = self.y_test
            total_frame["predicted"] = np.round(pred)
            incorrect = total_frame[total_frame["actual"] != total_frame["predicted"]]
            incorrect.to_csv(
                os.path.join(MODELS_PREDICTION_PATH, f"{obj_type}_{self.src_org_name}_{self.dst_org_name}_incorrect.csv"),
                index=False)
        return auc

    def get_previous_xgboost_score(self):
        data1 = pd.read_csv(os.path.join(MODELS_CROSS_ORG_TABELS, 'xgboost_17_11_2021 23_26_27.csv'), index_col=0)
        return data1.loc[self.src_org_name][self.dst_org_name]
