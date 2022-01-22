import xgboost as xgb
from keras import Model
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, Input
from keras.regularizers import l1

from src.models.training.base_model import BaseTrainObj
from src.models.training.xgb_model import XgboostTrainObj


def network_model(shape):
    x = Input(shape=(shape,), name="input")
    ann_dense1 = Dense(100, activation='tanh', name='dense_100')(x)
    ann_dense2 = Dense(50, activation='tanh', kernel_constraint=maxnorm(3), activity_regularizer=l1(0.001),
                       kernel_regularizer=l1(0.001), name='dense_50')(ann_dense1)
    ann_dropout = Dropout(rate=0.5, name='dropout')(ann_dense2)
    ann_dense3 = Dense(20, activation='tanh', name='dense_20')(ann_dropout)
    ann_output = Dense(1, activation='sigmoid', name='output')(ann_dense3)
    model = Model(x, ann_output, name="ann_model")
    return model


XGBS_PARAMS = {
    "objective": ["binary:hinge"],
    "booster": ["gbtree"],
    "eta": [0.1],
    'gamma': [0.5],
    'max_depth': range(2, 4, 2),
    'min_child_weight': [1],
    'subsample': [0.6],
    'colsample_bytree': [0.6],
    "lambda": [1],
    "n_jobs": [-1],
    "verbosity": 0,
    "silent": True,
}


def get_models_dict(shape):
    base_model = BaseTrainObj()
    base_model.set_model(network_model(shape))
    xbg_model = XgboostTrainObj()
    xbg_model.set_model(xgb.XGBClassifier(kwargs=XGBS_PARAMS))
    model_dict = {'base': base_model, 'xgb': xbg_model}
    return model_dict
