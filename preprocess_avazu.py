import numpy as np
import pandas as pd
import random, json
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


root_path = r''

avazu_file_path = os.path.join(root_path, "Avazu.csv")
vocab_file = os.path.join(root_path, 'embedding_dicts/vocabulary_size_10M.json')
des_file = os.path.join(root_path, '_.csv')

df = pd.read_table(avazu_file_path, sep=',')

df["hour"] = pd.to_datetime(df["hour"], format="%y%m%d%H")
df["actual_hour_day"]= df["hour"].dt.hour
df['day'] = df['hour'].dt.day

features = df.columns.tolist()
df[features] = df[features].fillna('',)
print (features)

# label encoding for categorical features

for feat in features:
    lbe = LabelEncoder() # encode target labels with value between 0 and n_classes-1.
    df.loc[:,feat] = lbe.fit_transform(df[feat]) # fit label encoder and return encoded label
    df.loc[:,feat] = df[feat].astype(np.int32) # convert from float64 to float32

df.drop(['id', 'hour'], axis=1, inplace=True)
df.rename(columns={'click': 'y',
                   'actual_hour_day': 'hour'}, inplace=True, errors='raise')

# Define the new column order
new_order = ['y', 'banner_pos', 'C1', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'hour', 'day', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model']
df = df.loc[:, new_order]

# Define the new column names
new_names = ['y'] + ['C{}'.format(i) for i in range(1, 24)]
df.columns = new_names

# Dict of dense features. 
vocab_dict = {}
for feat in ['C{}'.format(i) for i in range(1, 24)]:
    vocab_dict[feat] = df[feat].nunique()
print(vocab_dict)
with open(vocab_file, 'wb') as des:
    des.write(json.dumps(vocab_dict, ensure_ascii=False, indent=2).encode('utf-8'))

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Save pre-processed data. 
df.to_csv(path_or_buf=des_file, index=False)
