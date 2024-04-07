import numpy as np
import os
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import polars as pl


def column_check(data1, data2):
    return set(data1.columns) == set(data1.columns).intersection(set(data2.columns))


def merge_duplicate_group_cols(data):
    if 'num_group1_x' in data.columns:
        data = data.with_columns(
            pl.col('num_group1_x').fill_null(pl.col('num_group1_y')).alias('num_group1')
        ).drop(['num_group1_x', 'num_group1_y'])

    if 'num_group2_x' in data.columns:
        data = data.with_columns(
            pl.col('num_group2_x').fill_null(pl.col('num_group2_y')).alias('num_group2')
        ).drop(['num_group2_x', 'num_group2_y'])

    return data


def concat_and_merge(base_data, path_list, base_path):
    data = pl.concat([pl.read_csv(base_path + file) for file in path_list], how='diagonal_relaxed')
    data = data.drop_nulls('case_id').unique(subset='case_id', keep='first')
    base_data = base_data.join(data, on='case_id', how='left')

    return merge_duplicate_group_cols(base_data)


def get_file_names(file_names, keyword):
    return [x for x in file_names if keyword in x]


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

base = pl.read_csv(files_path + base_file[0])
base = concat_and_merge(base, applprev_files, files_path)
base = concat_and_merge(base, credit_a1_files, files_path)
base = concat_and_merge(base, credit_a2_files, files_path)
base = concat_and_merge(base, credit_b_files, files_path)
base = concat_and_merge(base, static0_files, files_path)

for file in rest_of_files:
    data = pl.scan_csv(files_path + file)
    data = data.drop_nulls('case_id').unique(subset='case_id', keep='first')
    base = base.join(data, on='case_id', how='left')
    base = merge_duplicate_group_cols(base)

print(base.select(pl.col("case_id").n_unique()))
null_counts = base.select(pl.all().null_count() / base.height)
missing_cols = null_counts[null_counts > 0.3].keys().to_list()

base = base.drop(missing_cols)

columns_to_fit = [col for col in base.columns if col not in ['case_id', 'target', 'num_group1', 'date_decision']]

object_cols = base[columns_to_fit].select_dtypes(include=['object']).columns.to_list()

encoders = {}
for col in object_cols:
    le = LabelEncoder()
    base = base.with_column(pl.col(col).map_encoding(lambda x: le.fit_transform([x])).alias(col))
    encoders[col] = le

lgb = LGBMClassifier(n_estimators=100)
lgb.fit(base[columns_to_fit].to_pandas(), base['target'].to_pandas(), categorical_feature=object_cols)

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

base = pl.scan_csv(files_path + base_file[0])
base = concat_and_merge(base, applprev_files, files_path)
base = concat_and_merge(base, credit_a1_files, files_path)
base = concat_and_merge(base, credit_a2_files, files_path)
base = concat_and_merge(base, credit_b_files, files_path)
base = concat_and_merge(base, static0_files, files_path)

for file in rest_of_files:
    data = pl.scan_csv(files_path + file)
    data = data.drop_nulls('case_id').unique(subset='case_id', keep='first')
    base = base.join(data, on='case_id', how='left')
    base = merge_duplicate_group_cols(base)

for col in object_cols:
    base = base.with_column(
        pl.col(col).map_encoding(
            lambda s: 'unknown' if s not in encoders[col].classes_ else s,
            encode_unknown_as_null=True,
        ).fill_null(pl.lit('unknown'), null_encoding='unknown').map_encoding(encoders[col].transform)
    )

submission = base[['case_id']]
submission = submission.with_column(
    pl.lit(lgb.predict_proba(base[columns_to_fit].to_pandas()))[:, 1].alias('score')
)
submission = submission.set_index('case_id')
submission.write_csv('./submission.csv')