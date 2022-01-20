import os

import numpy as np
import pandas as pd
import sklearn.model_selection as sk

from constants import *


def stratify_train_test_split(df, test_size, random_state):
    # Change: all the unique miRNA were put in the test set
    uniques_mirna = df[df.groupby("microRNA_name").microRNA_name.transform(len) == 1]
    non_uniques_mirna = df[df.groupby("microRNA_name").microRNA_name.transform(len) > 1]
    # dealing with the non_uniques_mirna
    non_uniques_train, non_uniques_test = sk.train_test_split(non_uniques_mirna, test_size=test_size,
                                                              random_state=random_state,
                                                              stratify=non_uniques_mirna["microRNA_name"])

    train = pd.concat([non_uniques_train])
    test = pd.concat([non_uniques_test, uniques_mirna])
    return train, test


def distribution_split(df, test_size):
    mirna_dist_cumsum = df["microRNA_name"].value_counts().cumsum()
    train_size = 1 - test_size
    min_train_size = round(df.shape[0] * train_size)
    tmp_test = df[df["microRNA_name"].isin(mirna_dist_cumsum[mirna_dist_cumsum > min_train_size].index)]
    tmp_train = df[~df["microRNA_name"].isin(tmp_test.index)]
    print(f"Total shape: {df.shape},train shape: {tmp_train.shape},test shape: {tmp_test.shape} ")
    return tmp_train, tmp_test


def get_data_from_file(data_file_name, test_size, remove_hot_paring, only_most_important, dist_split):
    pos_file_path = os.path.join(EXTERNAL_PATH, "{0}_pos.csv".format(data_file_name))
    pos = pd.read_csv(pos_file_path, index_col=0)
    pos.insert(0, "label", 1)
    neg_file_path = os.path.join(EXTERNAL_PATH, "{0}_neg.csv".format(data_file_name))
    neg = pd.read_csv(neg_file_path, index_col=0)
    neg.insert(0, "label", 0)
    col_to_remove = ['Source', 'Organism', 'number of reads']
    pos.drop(col_to_remove, axis=1, inplace=True)

    if remove_hot_paring:
        all_col_except_hot = [f for f in pos.columns if not str(f).startswith("HotPairing")]
        pos = pos[all_col_except_hot]
    col = [c for c in pos.columns if c in neg.columns]
    pos = pos[col]
    neg = neg[col]
    random_state = np.random.randint(15, 30)
    if dist_split:
        pos_train, pos_test = distribution_split(pos, test_size)
        neg_train, neg_test = distribution_split(neg, test_size)
    else:
        pos_train, pos_test = stratify_train_test_split(pos, test_size, random_state)
        neg_train, neg_test = stratify_train_test_split(neg, test_size, random_state)
    pos_train = pos_train.append(neg_train, ignore_index=True)
    pos_test = pos_test.append(neg_test, ignore_index=True)
    train = pos_train.reindex(np.random.RandomState(seed=random_state).permutation(pos_train.index))
    test = pos_test.reindex(np.random.RandomState(seed=random_state).permutation(pos_test.index))
    if only_most_important:
        train = train[IMPORTANT_FEATURES + SEQUANCE_FEATURES]
        test = test[IMPORTANT_FEATURES + SEQUANCE_FEATURES]
    return train, test


def create_train_dataset(org_name, remove_hot_paring, only_most_important, dist_split):
    print("\n---Reading miRNA external data---")
    train, test = get_data_from_file(org_name, 0.2, remove_hot_paring, only_most_important, dist_split)
    print("Training set shape is {0} and test is {1}\n".format(train.shape, test.shape))
    train.to_csv(DATA_PATH / f"{org_name}_train.csv", index=False)
    print("---Train dataset was created---\n")
    test.to_csv(DATA_PATH / f"{org_name}_test.csv", index=False)
    print("---Test dataset was created---\n")
