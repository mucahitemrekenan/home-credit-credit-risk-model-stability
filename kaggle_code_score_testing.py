import os
import gc
from glob import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

import joblib
from tqdm import tqdm
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier, DMatrix

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

print('import done')


class VotingModel(BaseEstimator, RegressorMixin):
	def __init__(self, estimators):
		super().__init__()
		self.estimators = estimators

	def fit(self, X, y=None):
		return self

	def predict(self, X):
		y_preds = [estimator.predict(X) for estimator in self.estimators]
		return np.mean(y_preds, axis=0)

	def predict_proba(self, X):
		y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
		return np.mean(y_preds, axis=0)


class Pipeline:
	@staticmethod
	def set_table_dtypes(df):
		for col in df.columns:
			if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
				df = df.with_columns(pl.col(col).cast(pl.Int64))
			elif col in ["date_decision"]:
				df = df.with_columns(pl.col(col).cast(pl.Date))
			elif col[-1] in ("P", "A"):
				df = df.with_columns(pl.col(col).cast(pl.Float64))
			elif col[-1] in ("M",):
				df = df.with_columns(pl.col(col).cast(pl.String))
			elif col[-1] in ("D",):
				df = df.with_columns(pl.col(col).cast(pl.Date))

		return df

	@staticmethod
	def handle_dates(df):
		for col in df.columns:
			if col[-1] in ("D",):
				df = df.with_columns(pl.col(col) - pl.col("date_decision"))
				df = df.with_columns(pl.col(col).dt.total_days())

		df = df.drop("date_decision", "MONTH")

		return df

	@staticmethod
	def filter_cols(df):
		for col in df.columns:
			if col not in ["target", "case_id", "WEEK_NUM"]:
				isnull = df[col].is_null().mean()

				if isnull > 0.95:
					df = df.drop(col)

		for col in df.columns:
			if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
				freq = df[col].n_unique()

				if (freq == 1) | (freq > 200):
					df = df.drop(col)

		return df


class Aggregator:
	@staticmethod
	def num_expr(df):
		cols = [col for col in df.columns if col[-1] in ("P", "A")]

		expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

		return expr_max

	@staticmethod
	def date_expr(df):
		cols = [col for col in df.columns if col[-1] in ("D",)]

		expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

		return expr_max

	@staticmethod
	def str_expr(df):
		cols = [col for col in df.columns if col[-1] in ("M",)]

		expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

		return expr_max

	@staticmethod
	def other_expr(df):
		cols = [col for col in df.columns if col[-1] in ("T", "L")]

		expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

		return expr_max

	@staticmethod
	def count_expr(df):
		cols = [col for col in df.columns if "num_group" in col]

		expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

		return expr_max

	@staticmethod
	def get_exprs(df):
		exprs = Aggregator.num_expr(df) + \
				Aggregator.date_expr(df) + \
				Aggregator.str_expr(df) + \
				Aggregator.other_expr(df) + \
				Aggregator.count_expr(df)

		return exprs


def read_file(path, depth=None):
	df = pl.read_parquet(path)
	df = df.pipe(Pipeline.set_table_dtypes)

	if depth in [1, 2]:
		df = df.group_by("case_id").agg(Aggregator.get_exprs(df))

	return df


def read_files(regex_path, depth=None):
	chunks = []
	for path in glob(str(regex_path)):
		chunks.append(pl.read_parquet(path).pipe(Pipeline.set_table_dtypes))

	df = pl.concat(chunks, how="vertical_relaxed")
	if depth in [1, 2]:
		df = df.group_by("case_id").agg(Aggregator.get_exprs(df))

	return df


def feature_eng(df_base, depth_0, depth_1, depth_2):
	df_base = (
		df_base
		.with_columns(
			month_decision=pl.col("date_decision").dt.month(),
			weekday_decision=pl.col("date_decision").dt.weekday(),
		)
	)

	for i, df in enumerate(depth_0 + depth_1 + depth_2):
		df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")

	df_base = df_base.pipe(Pipeline.handle_dates)

	return df_base


def to_pandas(df_data, cat_cols=None):
	df_data = df_data.to_pandas()

	if cat_cols is None:
		cat_cols = list(df_data.select_dtypes("object").columns)

	df_data[cat_cols] = df_data[cat_cols].astype("category")

	return df_data, cat_cols


def reduce_mem_usage(df):
	""" iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
	start_mem = df.memory_usage().sum() / 1024 ** 2
	print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

	for col in df.columns:
		col_type = df[col].dtype
		if str(col_type) == "category":
			continue

		if col_type != object:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)
		else:
			continue
	end_mem = df.memory_usage().sum() / 1024 ** 2
	print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
	print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

	return df


print('functions done')

ROOT = Path("D:/projects\home-credit-credit-risk-model-stability")
TRAIN_DIR = ROOT / "parquet_files" / "train"
TEST_DIR = ROOT / "parquet_files" / "test"

data_store = {
	"df_base": read_file(TRAIN_DIR / "train_base.parquet"),
	"depth_0": [
		read_file(TRAIN_DIR / "train_static_cb_0.parquet"),
		read_files(TRAIN_DIR / "train_static_0_*.parquet"),
	],
	"depth_1": [
		read_files(TRAIN_DIR / "train_applprev_1_*.parquet", 1),
		read_file(TRAIN_DIR / "train_tax_registry_a_1.parquet", 1),
		read_file(TRAIN_DIR / "train_tax_registry_b_1.parquet", 1),
		read_file(TRAIN_DIR / "train_tax_registry_c_1.parquet", 1),
		read_file(TRAIN_DIR / "train_credit_bureau_b_1.parquet", 1),
		read_file(TRAIN_DIR / "train_other_1.parquet", 1),
		read_file(TRAIN_DIR / "train_person_1.parquet", 1),
		read_file(TRAIN_DIR / "train_deposit_1.parquet", 1),
		read_file(TRAIN_DIR / "train_debitcard_1.parquet", 1),
	],
	"depth_2": [
		read_file(TRAIN_DIR / "train_credit_bureau_b_2.parquet", 2),
	]
}

df_train = feature_eng(**data_store)
print("train data shape:\t", df_train.shape)
df_train = df_train.pipe(Pipeline.filter_cols)
print("train data shape:\t", df_train.shape)
df_train, cat_cols = to_pandas(df_train)
df_train = reduce_mem_usage(df_train)
del data_store
gc.collect()
print('data read feat eng done')

cols_to_drop = ['lastrejectreason_759M',
				'maritalst_893M',
				'birthdate_574D',
				'max_empl_industry_691L',
				'max_rejectreasonclient_4145042M',
				'max_education_927M',
				'max_dateactivated_425D',
				'lastrejectcommoditycat_161M',
				'max_empladdr_district_926M',
				'max_recorddate_4527225D',
				'education_88M',
				'isdebitcard_729L',
				'max_empladdr_zipcode_114M',
				'requesttype_4525192L',
				'max_isdebitcard_527L',
				'max_inittransactioncode_279L',
				'max_approvaldate_319D',
				'max_housetype_905L',
				'max_empl_employedtotal_800L',
				'max_creationdate_885D',
				'cardtype_51L',
				'max_education_1138M',
				'lastrejectreasonclient_4145040M',
				'lastdelinqdate_224D',
				'max_postype_4733339M',
				'max_relationshiptoclient_642T',
				'max_contaddr_matchlist_1032L',
				'max_relationshiptoclient_415T',
				'paytype_783L',
				'max_isbidproduct_390L',
				'max_familystate_447L',
				'lastrejectdate_50D',
				'max_dtlastpmt_581D',
				'max_cancelreason_3545846M',
				'max_rejectreason_755M',
				'bankacctype_710L',
				'max_contaddr_smempladdr_334L',
				'max_credacc_status_367L',
				'lastcancelreason_561M',
				'lastrejectcommodtypec_5251769M',
				'max_remitter_829L',
				'typesuite_864L',
				'dtlastpmtallstes_4499206D',
				'education_1103M']

best_features = ['disbursedcredamount_1113A',
				 'maxlnamtstart6m_4525199A',
				 'credamount_770A',
				 'thirdquarter_1082L',
				 'price_1097A',
				 'maininc_215A',
				 'mindbddpdlast24m_3658935P',
				 'lastactivateddate_801D',
				 'max_mainoccupationinc_384A',
				 'max_amount_4527230A',
				 'maxannuity_159A',
				 'avgoutstandbalancel6m_4187114A',
				 'max_pmtnum_8L',
				 'fourthquarter_440L',
				 'dateofbirth_337D',
				 'datelastunpaid_3546854D',
				 'lastapprcommoditycat_1041M',
				 'pmtaverage_4527227A',
				 'lastapprdate_640D',
				 'days360_512L',
				 'maxdbddpdlast1m_3658939P',
				 'lastapprcredamount_781A',
				 'totinstallast1m_4525188A',
				 'datelastinstal40dpd_247D',
				 'max_annuity_853A',
				 'mobilephncnt_593L',
				 'lastrejectcredamount_222A',
				 'firstquarter_103L',
				 'pmtnum_254L',
				 'max_firstnonzeroinstldate_307D',
				 'max_birth_259D',
				 'maxdpdinstldate_3546855D',
				 'totalsettled_863A',
				 'annuity_780A',
				 'max_amount_4917619A',
				 'max_byoccupationinc_3656910L',
				 'firstclxcampaign_1125D',
				 'pctinstlsallpaidearl3d_427L',
				 'inittransactionamount_650A',
				 'pctinstlsallpaidlate1d_3546856L',
				 'maxdbddpdtollast12m_3658940P',
				 'avginstallast24m_3658937A',
				 'max_num_group1_3',
				 'validfrom_1069D',
				 'pmtssum_45A',
				 'firstdatedue_489D',
				 'max_pmtamount_36A',
				 'datefirstoffer_1144D',
				 'maxinstallast24m_3658928A',
				 'numinstlswithdpd10_728L',
				 'max_mainoccupationinc_437A',
				 'numincomingpmts_3546848L',
				 'max_employedfrom_700D',
				 'lastapplicationdate_877D',
				 'amtinstpaidbefduel24m_4187115A',
				 'max_dtlastpmtallstes_3545839D',
				 'max_empl_employedfrom_271D',
				 'eir_270L',
				 'avgdbddpdlast3m_4187120P',
				 'applicationscnt_867L',
				 'secondquarter_766L',
				 'max_credamount_590A']

all_cols = df_train.columns.difference(["target", "case_id", "WEEK_NUM"])
p_cols = [x for x in all_cols if
		  x.endswith('P') and x not in best_features and df_train[x].dtype not in ['object', 'category']]
a_cols = [x for x in all_cols if
		  x.endswith('A') and x not in best_features and df_train[x].dtype not in ['object', 'category']]
d_cols = [x for x in all_cols if
		  x.endswith('D') and x not in best_features and df_train[x].dtype not in ['object', 'category']]

numerical_cols = df_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()

numerical_cols = list(set(numerical_cols) - set(["target", "case_id", "WEEK_NUM"]))

boolean_cols = list()
for col in categorical_cols:
	categories = df_train[col].cat.categories
	if True in categories or False in categories:
		boolean_cols.extend([col])

categorical_cols = list(set(categorical_cols) - set(boolean_cols))

num_imputer = SimpleImputer(strategy='mean')  # or median
cat_imputer = SimpleImputer(strategy='most_frequent')
bool_imputer = SimpleImputer(strategy='most_frequent')

df_train[numerical_cols] = num_imputer.fit_transform(df_train[numerical_cols])
df_train[categorical_cols] = cat_imputer.fit_transform(df_train[categorical_cols])
df_train[boolean_cols] = bool_imputer.fit_transform(df_train[boolean_cols])

pca_models = dict()
scalers = dict()
pca_col_names = dict()
for cols, name in zip([p_cols, a_cols, d_cols], ['p', 'a', 'd']):
	length = round(len(cols) / 10)
	if length < 1:
		length = 1

	scaler = StandardScaler()
	pca = PCA(n_components=length)

	X_scaled = scaler.fit_transform(df_train[cols])
	principal_components = pca.fit_transform(X_scaled)
	pca_cols = [f'{name}_{x}' for x in range(length)]
	principal_components = pd.DataFrame(principal_components, columns=pca_cols)

	df_train.drop(cols, axis=1, inplace=True)
	df_train = pd.concat([df_train, principal_components], axis=1)

	pca_col_names[name] = pca_cols
	scalers[name] = scaler
	pca_models[name] = pca

df_train = df_train.drop(columns=cols_to_drop, errors='ignore')
X = df_train.drop(columns=["target", "case_id", "WEEK_NUM"])
y = df_train["target"]
weeks = df_train["WEEK_NUM"]
cols_to_fit = X.columns
cat_cols = list(set(cols_to_fit).intersection(set(cat_cols)))
print('training data ready')
del df_train
print('df_train deleted')

cv = StratifiedGroupKFold(n_splits=5, shuffle=False)

params = {
	"boosting_type": "gbdt",
	"objective": "binary",
	"metric": "auc",
	"max_depth": 8,
	"learning_rate": 0.05,
	"n_estimators": 1000,
	"colsample_bytree": 0.8,
	"colsample_bynode": 0.8,
	"verbose": -1,
	"random_state": 42,
	"device": "cpu"
}
fitted_models = []
cat_cols_lgb = X.select_dtypes(include=['object', 'category']).columns.tolist()

for column in cat_cols_lgb:
	X[column] = X[column].astype('category')

for idx_train, idx_valid in tqdm(cv.split(X, y, groups=weeks)):
	model = LGBMClassifier(**params)
	model.fit(
		X.iloc[idx_train], y.iloc[idx_train],
		eval_set=[(X.iloc[idx_valid], y.iloc[idx_valid])], categorical_feature=cat_cols_lgb
	)
	fitted_models.append(model)
model = VotingModel(fitted_models)
print('all models trained')
del X, y
gc.collect()
print('data deleted')


data_store = {
	"df_base": read_file(TEST_DIR / "test_base.parquet"),
	"depth_0": [
		read_file(TEST_DIR / "test_static_cb_0.parquet"),
		read_files(TEST_DIR / "test_static_0_*.parquet"),
	],
	"depth_1": [
		read_files(TEST_DIR / "test_applprev_1_*.parquet", 1),
		read_file(TEST_DIR / "test_tax_registry_a_1.parquet", 1),
		read_file(TEST_DIR / "test_tax_registry_b_1.parquet", 1),
		read_file(TEST_DIR / "test_tax_registry_c_1.parquet", 1),
		read_file(TEST_DIR / "test_credit_bureau_b_1.parquet", 1),
		read_file(TEST_DIR / "test_other_1.parquet", 1),
		read_file(TEST_DIR / "test_person_1.parquet", 1),
		read_file(TEST_DIR / "test_deposit_1.parquet", 1),
		read_file(TEST_DIR / "test_debitcard_1.parquet", 1),
	],
	"depth_2": [
		read_file(TEST_DIR / "test_credit_bureau_b_2.parquet", 2),
	]
}

df_test = feature_eng(**data_store)
print("test data shape:\t", df_test.shape)
#df_test = df_test.select([col for col in cols_to_fit if col != "target"])
print("test data shape:\t", df_test.shape)
df_test, cat_cols = to_pandas(df_test, cat_cols)
df_test = reduce_mem_usage(df_test)
del data_store
gc.collect()


df_test[numerical_cols] = num_imputer.transform(df_test[numerical_cols])
df_test[categorical_cols] = cat_imputer.transform(df_test[categorical_cols])
df_test[boolean_cols] = bool_imputer.transform(df_test[boolean_cols])

for cols, name in zip([p_cols, a_cols, d_cols], ['p', 'a', 'd']):

	pca = pca_models[name]
	scaler = scalers[name]

	X_scaled = scaler.transform(df_test[cols])
	principal_components = pca.transform(X_scaled)

	principal_components = pd.DataFrame(principal_components, columns=pca_col_names[name])

	df_test.drop(cols, axis=1, inplace=True)
	df_test = pd.concat([df_test, principal_components], axis=1)


# X_test = df_test.drop(columns=["WEEK_NUM"])
X_test = df_test.set_index("case_id")
X_test = df_test[cols_to_fit]

for column in cat_cols_lgb:
	X_test[column] = X_test[column].astype('category')

y_pred = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

df_subm = pd.read_csv(ROOT / "sample_submission.csv")
df_subm = df_subm.set_index("case_id")

df_subm["score"] = y_pred
df_subm.to_csv("submission.csv")
