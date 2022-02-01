import collections

import pandas as pd
from sklearn.model_selection import train_test_split

from constants import *


class DataOrg:

    def __init__(self, dataset_name):
        self.data = pd.DataFrame()
        self.v_data = None
        self.label = None
        self.v_label = None
        self.is_train = None
        self.dataset_name = dataset_name
        self.feature_names = None

    def __len__(self):
        if self.feature_names is None:
            raise ValueError('get size of an empty dataset')
        else:
            return len(self.feature_names)

    def __sizeof__(self):
        return self.data.shape

    def load(self, datasets, is_train, val_ratio=0):
        self.is_train = is_train
        for d in datasets:
            self.get_dataset(d)
        self.label = self.data['label']
        self.data = self.data.drop(FEATURES_TO_DROP, axis=1).fillna(0).astype("float")
        self.feature_names = list(self.data.columns)
        if val_ratio != 0:
            self.data, self.v_data, self.label, self.v_label = train_test_split(self.data, self.label,
                                                                                test_size=val_ratio,
                                                                                random_state=42)
        return self

    def get_dataset(self, org_name):
        df = pd.read_csv(DATA_PATH / f"{org_name}_{self.is_train}.csv", index_col=False)
        self.data = self.data.append(df)


class ScoreObj:

    def __init__(self, model_list, metrics):
        self.scores = collections.defaultdict(dict)
        self.metrics = metrics
        self.model_list = model_list
        for model in model_list:
            for metric in metrics:
                self.scores[model][metric] = collections.defaultdict(list)

    def add_score(self, model_name, score_dict, key, fill_empty=False):
        for metric in self.metrics:
            if fill_empty:
                self.scores[model_name][metric][key].append(0)
            else:
                self.scores[model_name][metric][key].append(round(score_dict[metric], 2))

    def index_to_col(self, df):
        df2 = df.reset_index(level=0)
        tmp = df2['index'].str.split('_', 1, expand=True)
        df2.insert(0, 'src_org', tmp[0])
        df2.insert(1, 'dst_org', tmp[1])
        df2 = df2.drop(['index'], axis=1)
        return df2

    def save_results(self, folder_name, header=None, keep_index=True):
        for model in self.model_list:
            for metric in self.metrics:
                data = pd.DataFrame.from_dict(self.scores[model][metric], orient='index')
                if not keep_index:
                    data = self.index_to_col(data)
                if header is not None:
                    try:
                        data.columns = header
                    except:
                        print(f"problem {model} {metric}")
                data.to_csv(folder_name / f"{model}_{metric}.csv",index=keep_index)
