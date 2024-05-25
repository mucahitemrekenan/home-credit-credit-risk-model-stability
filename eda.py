import numpy as np
import pandas as pd
import os
from lightgbm.sklearn import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import time
from tqdm import tqdm
from itertools import combinations
from data_prep import DataPreparation
import random


def search_column(col, data):
	result = [x for x in data.columns if col in x]
	print(result)
	return result[0]


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


def analyze(col, base):
	col = search_column(col, base)
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
		col = search_column(col, data)
		print('null_ratio', col, data[col].isnull().sum() / len(data))


base = DataPreparation().prepare_data()

null_counts = pd.DataFrame(data={'base': base.isnull().sum() / len(base),
								 'dpd': base[base['target'] == 1].isnull().sum() / len(base[base['target'] == 1])})
null_counts = null_counts.reset_index(names='features')

feature_defs = pd.read_csv('feature_definitions.csv')
feature_defs.columns = ['features', 'description']

null_counts = feature_defs.merge(null_counts, on='features', how='left')

total_nulls = base.isnull().mean()
feature_count_exists = False

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

	above_threshold_feature_counts = pd.DataFrame(list(above_threshold_feature_counts.items()),
												  columns=['features', 'count'])
	above_threshold_feature_counts.to_excel('above_threshold_feature_counts.xlsx', index=False)
	null_counts.to_excel('null_counts.xlsx', index=False)
else:
	above_threshold_feature_counts = pd.read_excel('above_threshold_feature_counts.xlsx')
	above_threshold_feature_counts.columns = ['features', 'count']
	null_counts = pd.read_excel('null_counts.xlsx')

above_threshold_feature_counts['null_ratio'] = above_threshold_feature_counts['count'] / len(
	above_threshold_feature_counts)
null_counts = above_threshold_feature_counts[['features']].merge(null_counts, on='features', how='left')

# actualdpd_943P
analyze('max_actualdpd_943P', base)
base = base.loc[base['max_actualdpd_943P'].notnull()]

# annuity_853A
analyze('annuity_853A', base)
sub_data = base[base['max_annuity_853A'].isnull()].isnull().mean()
base = base.loc[base['max_annuity_853A'].notnull()]

object_cols = base.select_dtypes(include=['category']).columns
object_cols = object_cols.tolist()

num_cols = base.select_dtypes(exclude=['category']).columns

# there are no high correlations between target and features
correlations = base[num_cols].corr()['target']

base_cols = ['case_id', 'WEEK_NUM', 'target']
category_cols = [col for col in object_cols if not col.endswith('D') and col not in base_cols]
category_data = base[category_cols]

category_data_counts = list()
for col in category_data:
	counts = category_data[col].value_counts(normalize=True).apply(lambda x: f'{x:.2f}').reset_index()
	category_data_counts.append(counts)
category_data_counts = transform_and_concat(category_data_counts)

date_data = base.filter(regex='D$')
date_col_definitions = feature_defs[feature_defs['features'].isin(date_data.columns)]

print('The ratio of identical rows of birthday columns to the overall data',
	  len(date_data[date_data['dateofbirth_337D'] == date_data['max_birth_259D']]) / len(date_data))

sub_data = base[['dateofbirth_337D', 'max_birth_259D']]

date_col_pairs = list(combinations(date_data.columns, 2))
pair_similarity_ratios = dict()
for pair in date_col_pairs:
	date_col1, date_col2 = pair
	pair_similarity_ratios[pair] = len(date_data[date_data[date_col1] == date_data[date_col2]]) / len(date_data)

pair_raitos_data = pd.DataFrame(pair_similarity_ratios.items())

col1, col2 = ('lastapplicationdate_877D', 'max_creationdate_885D')
check_null_ratio([col1], date_data)
check_null_ratio([col2], date_data)
sub_data = date_data[[col1, col2]]
cols_to_drop = ['max_creationdate_885D']

col1, col2 = ('lastapprdate_640D', 'max_approvaldate_319D')
check_null_ratio([col1], date_data)
check_null_ratio([col2], date_data)
sub_data = date_data[[col1, col2]]
cols_to_drop.extend(['max_approvaldate_319D'])

col1, col2 = ('lastactivateddate_801D', 'max_dateactivated_425D')
check_null_ratio([col1], date_data)
check_null_ratio([col2], date_data)
sub_data = date_data[[col1, col2]]
cols_to_drop.extend(['max_dateactivated_425D'])

col1, col2 = ('lastapplicationdate_877D', 'lastapprdate_640D')
check_null_ratio([col1], date_data)
check_null_ratio([col2], date_data)
sub_data = date_data.loc[base[base['target'] == 1].index, [col1, col2]]
sub_data = sub_data[sub_data['lastapprdate_640D'].notnull()]
len(sub_data[sub_data[col1] == sub_data[col2]]) / len(sub_data)

col1, col2 = ('datelastunpaid_3546854D', 'lastdelinqdate_224D')
check_null_ratio([col1], date_data)
check_null_ratio([col2], date_data)
sub_data = date_data[[col1, col2]]
sub_data = sub_data[sub_data['lastdelinqdate_224D'].notnull()]
len(sub_data[sub_data[col1] == sub_data[col2]]) / len(sub_data)
cols_to_drop.extend(['lastdelinqdate_224D'])

col1, col2 = ('max_dtlastpmt_581D', 'max_dtlastpmtallstes_3545839D')
check_null_ratio([col1], date_data)
check_null_ratio([col2], date_data)
sub_data = date_data[[col1, col2]]
sub_data = sub_data[sub_data['max_dtlastpmt_581D'].notnull()]
len(sub_data[sub_data[col1] == sub_data[col2]]) / len(sub_data)
cols_to_drop.extend(['max_dtlastpmt_581D'])

col1, col2 = ('dtlastpmtallstes_4499206D', 'max_dtlastpmtallstes_3545839D')
check_null_ratio([col1], date_data)
check_null_ratio([col2], date_data)
sub_data = date_data[[col1, col2]]
sub_data = sub_data[sub_data['dtlastpmtallstes_4499206D'].notnull()]
len(sub_data[sub_data[col1] == sub_data[col2]]) / len(sub_data)
cols_to_drop.extend(['dtlastpmtallstes_4499206D'])

col1, col2 = ('birthdate_574D', 'dateofbirth_337D')
check_null_ratio([col1], date_data)
check_null_ratio([col2], date_data)
sub_data = date_data[[col1, col2]]
sub_data = sub_data[sub_data['dateofbirth_337D'].notnull()]
len(sub_data[sub_data[col1] == sub_data[col2]]) / len(sub_data)
cols_to_drop.extend(['birthdate_574D'])

col1, col2 = ('lastapplicationdate_877D', 'lastrejectdate_50D')
check_null_ratio([col1], date_data)
check_null_ratio([col2], date_data)
sub_data = date_data[[col1, col2]]
cols_to_drop.extend(['lastrejectdate_50D'])

col1, col2 = ('responsedate_4527233D', 'max_recorddate_4527225D')
check_null_ratio([col1], date_data)
check_null_ratio([col2], date_data)
sub_data = date_data[[col1, col2]]
sub_data = sub_data[sub_data['responsedate_4527233D'].notnull()]
len(sub_data[sub_data[col1] == sub_data[col2]]) / len(sub_data)
cols_to_drop.extend(['max_recorddate_4527225D'])

col1, col2 = ('datelastunpaid_3546854D', 'maxdpdinstldate_3546855D')
check_null_ratio([col1], date_data)
check_null_ratio([col2], date_data)
sub_data = date_data[[col1, col2]]
sub_data = sub_data[sub_data['maxdpdinstldate_3546855D'].notnull()]
len(sub_data[sub_data[col1] == sub_data[col2]]) / len(sub_data)

cat_col_pairs = list(combinations(category_data.columns, 2))
pair_similarity_ratios = dict()
for pair in cat_col_pairs:
	col1, col2 = pair
	pair_similarity_ratios[pair] = len(
		category_data.loc[category_data[col1].astype(str) == category_data[col2].astype(str)]) / len(category_data)
pair_ratios_data = pd.DataFrame(pair_similarity_ratios.items())
pair_ratios_data.columns = ['pair', 'similarity_ratio']
pairs_above_threshold = pair_ratios_data[pair_ratios_data['similarity_ratio'] > 0.50]

for col1, col2 in pairs_above_threshold['pair']:
	null1 = len(category_data[col1].isnull()) / len(category_data)
	null2 = len(category_data[col2].isnull()) / len(category_data)

	if null1 > null2:
		cols_to_drop.extend([col1])
	elif null2 > null1:
		cols_to_drop.extend([col2])
	elif null1 == null2:
		if 'max' in col1 and 'max' in col2:
			cols_to_drop.extend([random.choice([col1, col2])])
		elif 'max' in col1:
			cols_to_drop.extend([col1])
		elif 'max' in col2:
			cols_to_drop.extend([col2])
		else:
			cols_to_drop.extend([random.choice([col1, col2])])
	else:
		pass

cols_to_drop = list(set(cols_to_drop))


