import os
import sys
from time import strftime, localtime, time
from datetime import date
import logging
import pandas as pd
import torch
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 1. Load all encoders
les = {}
for j in ['C{}'.format(str(i)) for i in range(1, 27)]:
    with open('embedding_dicts/' + j + ".json", 'r') as src:
        dic = json.load(src)
    les[j] = dic
print('Finished dic loading')

with open('embedding_dicts/vocabulary_size.json', 'r') as src:
    voca = json.load(src)

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]
data_column_names = ['label'] + dense_features + sparse_features

def process_training_data():
    f = 'dataset.csv'
    data = pd.read_csv('raw_data/criteo/' + f, names=data_column_names, delimiter="\t")
    # Sparse Feature
    data[sparse_features] = data[sparse_features].fillna('NaN', )
    for feat in sparse_features:
        print(f, feat)
        data[feat] = data[feat].apply(lambda x: les[feat].get(x, voca[feat]-1))
    # Dense Feature
    data[dense_features] = data[dense_features].fillna(0, )

    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # Save
    data.to_csv('raw_data/criteo/' + f, header=False, index=False)
    print(f)

def generate_vocabulary_size_dict(data_path):
    data, sparse_features, dense_features = load_data(data_path)
    data[sparse_features] = data[sparse_features].fillna('-1', )

    vocabulary_size_dict = {}
    for feat in sparse_features:
        lbe = LabelEncoder()   # Should build label encoder on Full data and Save it. 
        data[feat] = lbe.fit_transform(data[feat])
        vocabulary_size_dict[feat] = data[feat].nunique()
        # print(feat, vocabulary_size_dict[feat])
    return vocabulary_size_dict


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def encode_cate_features(data_path, des_path):
    """
    Load data from file
    :param data_path:
    :return:
    """
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    data_column_names = ['label'] + dense_features + sparse_features
    label_encoders = {}
    vocabulary_size_dict = {}

    for sparse in ['C' + str(i) for i in range(1, 27)]:
        time_ = time()
        data = pd.read_csv(data_path, names=data_column_names, usecols=[sparse])
        data = data.fillna('NaN', )

        print("load time: ", time() - time_)
        le = LabelEncoder()
        time_ = time()
        le.fit(data[sparse])

        keys = le.classes_
        values = le.transform(le.classes_)
        label_encoders[sparse] = dict(zip(keys, values))

        print("fit time: ", time() - time_)
        if not( len(label_encoders[sparse]) == data[sparse].nunique() ):
            print("wrong ", len(label_encoders[sparse]), data[sparse].nunique())

        vocabulary_size_dict[sparse] = max(len(label_encoders[sparse]), data[sparse].nunique())
        print(sparse, vocabulary_size_dict[sparse])
        with open(des_path + "{}.json".format(sparse), "w+") as des:
            json.dump(label_encoders[sparse], des, cls=NpEncoder)

    with open(des_path + "vocabulary_size.json", "w+") as des:
        json.dump(vocabulary_size_dict, des, cls=NpEncoder)

    # with open(des_path + "all_feature_encoders.json", "w+") as des:
    #     json.dump(label_encoders, des, cls=NpEncoder)

    print("Data processed successfully.")



# Useless method
# Test data are not labeled!!! 
def process_test_data():
    with open('embedding_dicts/vocabulary_size.json', 'r') as src:
        voca = json.load(src)

    data = pd.read_csv('test_file_path', names=data_column_names, delimiter="\t")
    print('Finish loading data')
    # Sparse Feature
    data[sparse_features] = data[sparse_features].fillna('NaN', )
    data[dense_features] = data[dense_features].fillna(0, )
    data.to_csv('raw_data/criteo/', header=False, index=False)

    print('Finish fill NaN')
    for feat in sparse_features:
        print(feat)
        data[feat] = data[feat].apply(lambda x: les[feat].get(x, voca[feat]-1))
    # Save
    data.to_csv('raw_data/criteo/test_file_path', header=False, index=False)

if __name__ == '__main__':
    process_training_data()
