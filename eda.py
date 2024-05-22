import numpy as np
import pandas as pd
import os
from lightgbm.sklearn import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import time
from tqdm import tqdm
from itertools import combinations

def column_check(data1, data2):
    return set(data1.columns) == set(data1.columns).intersection(data2.columns)


def timer(func):
    def wrapper(*args, **kwargs):
        file_name = args[1][0].replace('.csv', '')
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        exec_time = end_time - start_time
        if exec_time <= 60:
            print(f"Function {file_name} executed in {exec_time:.2f} seconds")
        else:
            print(f"Function {file_name} executed in {exec_time / 60:.2f} minutes")
        return result
    return wrapper


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


@timer
def concat_and_merge(base_data, path_list, base_path, rows, cols_to_merge=None):
    data = pd.DataFrame()
    for file in path_list:
        if cols_to_merge:
            # I specify whether read all columns or a given list of columns.
            data = pd.concat([data, pd.read_csv(base_path + file, usecols=['case_id'] + cols_to_merge, nrows=rows,
                                                low_memory=False)], axis=0)
        else:
            data = pd.concat([data, pd.read_csv(base_path + file, nrows=rows, low_memory=False)], axis=0)

    data.drop_duplicates(subset=['case_id'], keep='first', inplace=True)
    base_data = base_data.merge(data, on='case_id', how='left')

    return merge_duplicate_group_cols(base_data)


def search_features(cols_to_search, base_path):
    applprev = pd.read_csv(base_path + 'train_applprev_1_0.csv', nrows=1)
    credit_a1 = pd.read_csv(base_path + 'train_credit_bureau_a_1_0.csv', nrows=1)
    static0 = pd.read_csv(base_path + 'train_static_0_0.csv', nrows=1)
    rest = pd.DataFrame()
    for file in rest_of_files:
        rest = pd.concat([rest, pd.read_csv(base_path + file, nrows=1)], axis=0)

    appl_features = list(set(applprev.columns).intersection(set(cols_to_search)))
    credit_features = list(set(credit_a1.columns).intersection(set(cols_to_search)))
    static0_features = list(set(static0.columns).intersection(set(cols_to_search)))
    rest_features = list(set(rest.columns).intersection(set(cols_to_search)))

    return appl_features, credit_features, static0_features, rest_features
def get_file_names(file_names, keyword):
    return [x for x in file_names if keyword in x]


def is_include_features(path_list, base_path, feature_names: list) -> list:
    data = pd.DataFrame()
    for file in path_list:
        data = pd.concat([data, pd.read_csv(base_path + file, nrows=1, low_memory=False)], axis=0)
    return list(set(feature_names).intersection(set(data.columns)))


def analyze(col, base):
    print(base[col].value_counts(normalize=True).apply(lambda x: f"{x:.5f}"))
    print(f'{col} null ratio:', base[col].isnull().sum() / len(base))
    print(f'{col} null ratio in target==1:', base.loc[base['target'] == 1, col].isnull().sum() / len(base))


def transform_and_concat(dataframes):
    data = {'col1': [], 'col2': []}
    for df in dataframes:
        col0 = df.columns[0]
        col1 = df.columns[1]
        data['col1'].append(col0)
        data['col1'].extend(df[col0])
        data['col2'].append(col1)
        data['col2'].extend(df[col1])
    return pd.DataFrame(data)


def check_null_ratio(cols, data):
    for col in cols:
        print('null_ratio', col, data[col].isnull().sum() / len(data))


def analyze_col_pairs(cols, base_data, feature_defs):
    col1, col2 = cols
    check_null_ratio([col1, col2], base_data)
    print('dissimilar row count:', len(base_data[base_data[col1] != base_data[col2]]))
    print(feature_defs[feature_defs['features'].isin([col1, col2])].values)


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

start_time = time.time()
for file in rest_of_files:
    data = pd.read_csv(files_path + file, nrows=nrows, low_memory=False)
    data.drop_duplicates(subset=['case_id'], keep='first', inplace=True)
    base = base.merge(data, on='case_id', how='left')
    base = merge_duplicate_group_cols(base)
end_time = time.time()
print(f"Execution time rest of files: {end_time - start_time:.2f} seconds")
del data

base['date_decision'] = pd.to_datetime(base['date_decision'])

base['year'] = base['date_decision'].dt.year
base['month'] = base['date_decision'].dt.month
base['day'] = base['date_decision'].dt.day

base['date_decision'].value_counts().plot()
plt.show()

dpd_samples = base[base['target'] == 1].copy()
null_counts = pd.DataFrame(data={'base': base.isnull().sum() / len(base),
                                 'dpd': dpd_samples.isnull().sum() / len(dpd_samples)})
null_counts = null_counts.reset_index(names='features')

feature_defs = pd.read_csv('feature_definitions.csv')
feature_defs.columns = ['features', 'description']

null_counts = feature_defs.merge(null_counts, on='features', how='left')

total_nulls = base.isnull().mean()
feature_count_exists = True

if not feature_count_exists:
    null_threshold = 0.5
    null_flags = base.isnull()
    above_threshold_feature_counts = {}

    for col in tqdm(base.columns):
        null_indices = null_flags[col]
        subset_null_ratios = null_flags.loc[null_indices].mean().reset_index()
        subset_null_ratios.columns = ['features', col]
        null_counts = null_counts.merge(subset_null_ratios, on='features', how='left')
        above_threshold_feature_counts[col] = (subset_null_ratios[col] > null_threshold).sum()

    above_threshold_feature_counts = pd.DataFrame(list(above_threshold_feature_counts.items()), columns=['features', 'count'])
    above_threshold_feature_counts.to_excel('above_threshold_feature_counts.xlsx', index=False)
    null_counts.to_excel('null_counts.xlsx', index=False)
else:
    above_threshold_feature_counts = pd.read_excel('above_threshold_feature_counts.xlsx')
    above_threshold_feature_counts.columns = ['features', 'count']
    null_counts = pd.read_excel('null_counts.xlsx')

above_threshold_feature_counts['null_ratio'] = above_threshold_feature_counts['count'] / len(above_threshold_feature_counts)
null_counts = above_threshold_feature_counts[['features']].merge(null_counts, on='features', how='left')

# actualdpd_943P
analyze('actualdpd_943P', base)
base = base.loc[base['actualdpd_943P'].notnull()]

# annuity_853A
analyze('annuity_853A', base)
base = base.loc[base['annuity_853A'].notnull()]

total_nulls = base.isnull().mean().reset_index()
total_nulls.columns = ['features', 'null_ratio']
total_nulls = total_nulls.merge(feature_defs, on='features', how='left')

cols_to_hold = total_nulls.loc[total_nulls['null_ratio'] <= 0.2, 'features'].tolist()

base = base.drop(columns=base.columns.difference(cols_to_hold))

appl_features, credit_features, static0_features, rest_features = search_features(cols_to_hold, files_path)

object_cols = base[cols_to_hold].select_dtypes(include=['object']).columns
object_cols = object_cols.tolist()

# there are no high correlations between target and features
correlations = base.corr()['target']

object_data = base[object_cols].copy()
non_date_categorical_cols = [col for col in object_data.columns if not col.endswith('D')]
non_date_object_data = object_data[non_date_categorical_cols].copy()

value_counts_non_date_object_data = list()
for col in non_date_categorical_cols:
    counts = non_date_object_data[col].value_counts(normalize=True).apply(lambda x: f'{x:.2f}').reset_index()
    value_counts_non_date_object_data.append(counts)

value_counts_non_date_object_data = transform_and_concat(value_counts_non_date_object_data)

date_data = base.filter(regex='D$')

date_col_definitions = feature_defs[feature_defs['features'].isin(date_data.columns)].copy()

print('The ratio of identical rows of birthday columns to the overall data',
      len(date_data[date_data['dateofbirth_337D'] == date_data['birth_259D']]) / len(date_data))

date_col_pairs = list(combinations(date_data.columns, 2))
pair_similarity_ratios = dict()
for pair in date_col_pairs:
    date_col1, date_col2 = pair
    pair_similarity_ratios[pair] = len(date_data[date_data[date_col1] == date_data[date_col2]]) / len(date_data)

pair_raitos_data = pd.DataFrame(pair_similarity_ratios.items())

check_null_ratio('creationdate_885D', date_data)
check_null_ratio('lastapplicationdate_877D', date_data)

cols_to_drop = ['lastapplicationdate_877D']

check_null_ratio('dateofbirth_337D', date_data)
check_null_ratio('birth_259D', date_data)

cols_to_drop.extend(['dateofbirth_337D'])


check_null_ratio('lastapplicationdate_877D', date_data)
check_null_ratio('lastapprdate_640D', date_data)

# the ratios were different, nunique doesn't count nan values as unique. i will use '=' operator for similarity check
print('similar row ratio:', len(base[base[['lastapplicationdate_877D', 'lastapprdate_640D']].nunique(axis=1) == 1]) / len(base))
print('similar row ratio:', len(base[base['lastapplicationdate_877D'] == base['lastapprdate_640D']]) / len(base))

sub_data = base.loc[base[['lastapplicationdate_877D', 'lastapprdate_640D']].nunique(axis=1) == 1, ['lastapplicationdate_877D', 'lastapprdate_640D']]
sub_data2 = base.loc[base['lastapplicationdate_877D'] == base['lastapprdate_640D'], ['lastapplicationdate_877D', 'lastapprdate_640D']]

# I inspect the situation on target value between these columns.
sub_data = base.loc[base['target'] == 1, ['lastapplicationdate_877D', 'lastapprdate_640D']]
print(sub_data.isnull().sum() / len(sub_data))
sub_data2 = sub_data[sub_data.notnull().all(axis=1)]
print('similar row ratio:', len(sub_data2[sub_data2['lastapplicationdate_877D'] == sub_data2['lastapprdate_640D']]) / len(sub_data2))

cols_to_drop.extend(['lastapprdate_640D'])

check_null_ratio('firstnonzeroinstldate_307D', date_data)
check_null_ratio('firstdatedue_489D', date_data)

print('similar row ratio:', len(base[base['firstnonzeroinstldate_307D'] == base['firstdatedue_489D']]) / len(base))

object_col_pairs = list(combinations(non_date_categorical_cols, 2))
object_similarity_ratios = dict()
for pair in tqdm(object_col_pairs):
    col1, col2 = pair
    object_similarity_ratios[pair] = len(non_date_object_data[non_date_object_data[col1] == non_date_object_data[col2]]) / len(non_date_object_data)

object_raitos_data = pd.DataFrame(object_similarity_ratios.items())
object_raitos_data.to_excel('object_similarity_ratios.xlsx', index=False)

col1, col2 = ('status_219L', 'lastst_736L')
analyze_col_pairs([col1, col2], base, feature_defs)
sub_data = base.loc[base[col1] != base[col2], [col1, col2]]
condition = (base[col1] != base[col2]) & (base[col1] == 'D')
base.loc[condition, col1] = base.loc[condition, col2]
cols_to_drop.extend(['lastst_736L'])

col1, col2 = ('contaddr_matchlist_1032L', 'contaddr_smempladdr_334L')
analyze_col_pairs([col1, col2], base, feature_defs)
sub_data = base.loc[base[col1] != base[col2], [col1, col2]]
condition = (base[col1] != base[col2]) & (base[col1] == 'D')
base.loc[condition, col1] = base.loc[condition, col2]
cols_to_drop.extend(['contaddr_smempladdr_334L'])

analyze_col_pairs(('profession_152M', 'lastrejectcommodtypec_5251769M'), base, feature_defs)
sub_data = base.loc[base['profession_152M'] != base['lastrejectcommodtypec_5251769M'], ['profession_152M', 'lastrejectcommodtypec_5251769M']]
condition = (base['profession_152M'] != base['lastrejectcommodtypec_5251769M']) & (base['profession_152M'] == 'a55475b1')
base.loc[condition, 'profession_152M'] = base.loc[condition, 'lastrejectcommodtypec_5251769M']
cols_to_drop.extend(['lastrejectcommodtypec_5251769M'])
cols_to_drop.extend(['lastapprcommoditytypec_5251766M'])

analyze_col_pairs(('cancelreason_3545846M', 'lastcancelreason_561M'), base, feature_defs)
sub_data = base.loc[base['cancelreason_3545846M'] != base['lastcancelreason_561M'], ['cancelreason_3545846M', 'lastcancelreason_561M']]
condition = (base['cancelreason_3545846M'] != base['lastcancelreason_561M']) & (base['cancelreason_3545846M'] == 'a55475b1')
base.loc[condition, 'cancelreason_3545846M'] = base.loc[condition, 'lastcancelreason_561M']
cols_to_drop.extend(['lastcancelreason_561M'])

