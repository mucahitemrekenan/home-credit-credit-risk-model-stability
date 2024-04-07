import numpy as np
import pandas as pd
import os
from lightgbm.sklearn import LGBMClassifier
from sklearn.preprocessing import LabelEncoder


def column_check(data1, data2):
    return set(data1.columns) == set(data1.columns).intersection(data2.columns)


def merge_duplicate_group_cols(data):
    if 'num_group1_x' in data.columns:
        data.loc[:, 'num_group1_x'] = data.loc[:, 'num_group1_x'].fillna(data.loc[:, 'num_group1_y'])
        data.drop('num_group1_y', axis=1, inplace=True)
        data.rename(columns={'num_group1_x': 'num_group1'}, inplace=True)

    if 'num_group2_x' in data.columns:
        data.loc[:, 'num_group2_x'] = data.loc[:, 'num_group2_x'].fillna(data.loc[:, 'num_group2_y'])
        data.drop('num_group2_y', axis=1, inplace=True)
        data.rename(columns={'num_group2_x': 'num_group2'}, inplace=True)
    return data


def concat_and_merge(base_data, path_list, base_path, rows):
    data = pd.DataFrame()
    for file in path_list:
        data = pd.concat([data, pd.read_csv(base_path + file, nrows=rows, low_memory=False)], axis=0)

    data.drop_duplicates(subset=['case_id'], keep='first', inplace=True)
    base_data = base_data.merge(data, on='case_id', how='left')

    return merge_duplicate_group_cols(base_data)


def get_file_names(file_names, keyword):
    return [x for x in file_names if keyword in x]


def is_include_features(path_list, base_path, feature_names: list) -> set:
    data = pd.DataFrame()
    for file in path_list:
        data = pd.concat([data, pd.read_csv(base_path + file, nrows=1, low_memory=False)], axis=0)
    return set(feature_names).intersection(set(data.columns))


nrows = None
files_path = 'csv_files/train/'

files = os.listdir(files_path)

base_file = ['train_base.csv']
applprev_files = get_file_names(files, 'applprev')
credit_a1_files = get_file_names(files, 'credit_bureau_a_1')
credit_a2_files = get_file_names(files, 'credit_bureau_a_2')
credit_b_files = get_file_names(files, 'credit_bureau_b')
static0_files = get_file_names(files, 'static_0')
rest_of_files = set(files) - set(applprev_files + credit_a1_files + credit_a2_files + credit_b_files +
                                 static0_files + base_file)


base = pd.read_csv(files_path + base_file[0], nrows=nrows)
base = concat_and_merge(base, applprev_files, files_path, nrows)
base = concat_and_merge(base, credit_a1_files, files_path, nrows)
base = concat_and_merge(base, credit_a2_files, files_path, nrows)
base = concat_and_merge(base, credit_b_files, files_path, nrows)
base = concat_and_merge(base, static0_files, files_path, nrows)

for file in rest_of_files:
    data = pd.read_csv(files_path + file, nrows=nrows, low_memory=False)
    data.drop_duplicates(subset=['case_id'], keep='first', inplace=True)
    base = base.merge(data, on='case_id', how='left')
    base = merge_duplicate_group_cols(base)

del data

print(base['case_id'].nunique())
null_counts = base.apply(lambda x: x.isnull().sum() / len(x))
missing_cols = null_counts[null_counts > 0.3].index.tolist()

base.drop(columns=missing_cols, inplace=True)

columns_to_fit = base.columns.drop(['case_id', 'target', 'num_group1', 'date_decision']).tolist()

object_cols = base[columns_to_fit].select_dtypes(include=['object']).columns
object_cols = object_cols.tolist()

encoders = dict()
for col in object_cols:
    le = LabelEncoder()
    base[col] = le.fit_transform(base[col])
    encoders[col] = le

lgb = LGBMClassifier(n_estimators=100)
lgb.fit(base[columns_to_fit], base['target'], categorical_feature=object_cols)

feature_importances = pd.DataFrame({'feature': lgb.feature_name_,'imp_point': lgb.feature_importances_})
feature_importances.sort_values(by='imp_point', ascending=False, inplace=True)
best20_fetures = list(feature_importances['feature'].head(20))

del base

appl_features = is_include_features(applprev_files, files_path, best20_fetures)
credit_a1_features = is_include_features(credit_a1_files, files_path, best20_fetures)
credit_a2_features = is_include_features(credit_a2_files, files_path, best20_fetures)
credit_b_features = is_include_features(credit_b_files, files_path, best20_fetures)
static0_features = is_include_features(static0_files, files_path, best20_fetures)


#==============================================
# test
#==============================================

nrows = None
files_path = 'csv_files/test/'

files = os.listdir(files_path)

base_file = ['test_base.csv']
applprev_files = get_file_names(files, 'applprev')
credit_a1_files = get_file_names(files, 'credit_bureau_a_1')
credit_a2_files = get_file_names(files, 'credit_bureau_a_2')
credit_b_files = get_file_names(files, 'credit_bureau_b')
static0_files = get_file_names(files, 'static_0')
rest_of_files = set(files) - set(applprev_files + credit_a1_files + credit_a2_files + credit_b_files +
                                 static0_files + base_file)

base = pd.read_csv(files_path + base_file[0], nrows=nrows)
base = concat_and_merge(base, applprev_files, files_path, nrows)
base = concat_and_merge(base, credit_a1_files, files_path, nrows)
base = concat_and_merge(base, credit_a2_files, files_path, nrows)
base = concat_and_merge(base, credit_b_files, files_path, nrows)
base = concat_and_merge(base, static0_files, files_path, nrows)

for file in rest_of_files:
    data = pd.read_csv(files_path + file, nrows=nrows)
    data.drop_duplicates(subset=['case_id'], keep='first', inplace=True)
    base = base.merge(data, on='case_id', how='left')
    base = merge_duplicate_group_cols(base)

del data

for col in object_cols:
    base[col] = base[col].map(lambda s: 'unknown' if s not in encoders[col].classes_ else s)
    encoders[col].classes_ = np.append(encoders[col].classes_, 'unknown')  # Add 'unknown' to classes
    base[col] = encoders[col].transform(base[col])

submission = base[['case_id']]
submission.loc[:, 'score'] = lgb.predict_proba(base[columns_to_fit])[:, 1]
submission = submission.set_index('case_id')
submission.to_csv('./submission.csv')