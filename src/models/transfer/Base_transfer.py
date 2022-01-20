import os
from constants import *
from src.models.miRNA_transfer_subclass import api_model
import logging

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
from src.models.transfer.Transfer_obj import Transfer_obj


class BaseTransferObj(Transfer_obj):

    def __init__(self, org_name,aa1=False):
        Transfer_obj.__init__(self, org_name)
        self.l_model = api_model(MODEL_INPUT_SHAPE)
        print("set all layers to no trainable")
        for l in self.l_model.layers:
            print(l.name, l.trainable)
            l.trainable = False
        self.l_model.get_layer('dense_20').trainable = True
        self.l_model.get_layer('output').trainable = True

    def retrain_model(self, t_size, obj_type, trans_epochs,aa1=False):
        self.l_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        load_weights_path = os.path.join(MODELS_OBJECTS_PATH, f"{self.src_model_name}/")
        if aa1:
            load_weights_path = os.path.join(MODELS_OBJECTS_PATH, f"aa1_{self.src_model_name}/")
        self.l_model.load_weights(load_weights_path)
        return super(BaseTransferObj, self).retrain_model(t_size, 'base', trans_epochs,aa1)
