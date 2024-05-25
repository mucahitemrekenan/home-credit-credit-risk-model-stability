# %%
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


class DataPreparation:
	def __init__(self):
		self.df_train = None

	def prepare_data(self):
		ROOT = Path("D:/projects\home-credit-credit-risk-model-stability")
		TRAIN_DIR = ROOT / "parquet_files" / "train"

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

		self.df_train = feature_eng(**data_store)
		print("train data shape:\t", self.df_train.shape)
		self.df_train = self.df_train.pipe(Pipeline.filter_cols)
		print("train data shape:\t", self.df_train.shape)
		self.df_train, cat_cols = to_pandas(self.df_train)
		self.df_train = reduce_mem_usage(self.df_train)
		del data_store
		gc.collect()
		return self.df_train


