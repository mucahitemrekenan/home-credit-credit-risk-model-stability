import numpy as np
import pandas as pd
import os


def column_check(data1, data2):
    return set(data1.columns) == set(data1.columns).intersection(data2.columns)


def concat_and_merge(train_data, path_list, train_files_path, nrows):
    data = pd.DataFrame()
    for file in path_list:
        data = pd.concat([data, pd.read_csv(train_files_path + file, nrows=nrows, low_memory=False)], axis=0)

    data.drop_duplicates(subset=['case_id'], keep='first', inplace=True)
    return train_data.merge(data, on='case_id', how='left')


nrows = None
train_files_path = 'csv_files/train/'

train_files = os.listdir(train_files_path)
train = pd.read_csv(train_files_path + 'train_base.csv', nrows=nrows)

applprev_paths = ['train_applprev_1_0.csv', 'train_applprev_1_1.csv']
train = concat_and_merge(train, applprev_paths, train_files_path, nrows)

credit_a1_paths = ['train_credit_bureau_a_1_0.csv', 'train_credit_bureau_a_1_1.csv', 'train_credit_bureau_a_1_2.csv',
                   'train_credit_bureau_a_1_3.csv']
train = concat_and_merge(train, credit_a1_paths, train_files_path, nrows)

credit_a2_paths = ['train_credit_bureau_a_2_0.csv', 'train_credit_bureau_a_2_1.csv', 'train_credit_bureau_a_2_10.csv',
                   'train_credit_bureau_a_2_2.csv', 'train_credit_bureau_a_2_3.csv', 'train_credit_bureau_a_2_4.csv',
                   'train_credit_bureau_a_2_5.csv', 'train_credit_bureau_a_2_6.csv', 'train_credit_bureau_a_2_7.csv',
                   'train_credit_bureau_a_2_8.csv', 'train_credit_bureau_a_2_9.csv', 'train_credit_bureau_a_2_10.csv']
train = concat_and_merge(train, credit_a2_paths, train_files_path, nrows)

credit_b_paths = ['train_credit_bureau_b_1.csv', 'train_credit_bureau_b_2.csv']
train = concat_and_merge(train, credit_b_paths, train_files_path, nrows)

static_paths = ['train_static_0_0.csv', 'train_static_0_1.csv']
train = concat_and_merge(train, static_paths, train_files_path, nrows)

rest_of_files = ['train_debitcard_1.csv', 'train_deposit_1.csv',
                 'train_other_1.csv', 'train_person_1.csv', 'train_person_2.csv',
                 'train_static_cb_0.csv', 'train_tax_registry_a_1.csv', 'train_tax_registry_b_1.csv',
                 'train_tax_registry_c_1.csv']

for file in rest_of_files:
    data = pd.read_csv(train_files_path + file, nrows=nrows)
    data.drop_duplicates(subset=['case_id'], keep='first', inplace=True)
    train = train.merge(data, on='case_id', how='left')

del data

print(train['case_id'].nunique())
null_counts = train.apply(lambda x: x.isnull().sum() / len(x))
missing_cols = null_counts[null_counts > 0.4].index.tolist()
