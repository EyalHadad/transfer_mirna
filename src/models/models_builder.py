import xgboost as xgb
from keras import Model
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, Input
from keras.regularizers import l1
from tensorflow.keras import regularizers
from src.models.training.base_model import BaseTrainObj
from src.models.training.xgb_model import XgboostTrainObj


def network_model(shape):
    x = Input(shape=(shape,), name="input")
    _model = Dense(300, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5))(x)
    _model = Dropout(rate=0.6)(_model)
    _model = Dense(200,activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5))(_model)
    _model = Dropout(rate=0.6)(_model)
    _model = Dense(100, activation='relu')(_model)
    _model = Dropout(rate=0.6)(_model)
    _model = Dense(20, activation='relu')(_model)
    _model = Dense(1, activation='sigmoid')(_model)
    model = Model(x, _model, name="ann_model")
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
