import numpy as np
import pandas as pd
import os
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from xgboost import XGBClassifier


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


def get_file_names(file_names, keyword):
	return [x for x in file_names if keyword in x]


nrows = None
files_path = 'csv_files/train/'

files = os.listdir(files_path)

base_file = ['train_base.csv']
applprev_files = get_file_names(files, 'applprev_1')
credit_a1_files = get_file_names(files, 'credit_bureau_a_1')
credit_a2_files = get_file_names(files, 'credit_bureau_a_2')
credit_b_files = get_file_names(files, 'credit_bureau_b')
static0_files = get_file_names(files, 'static_0')
rest_of_files = set(files) - set(applprev_files + credit_a1_files + credit_a2_files + credit_b_files +
								 static0_files + base_file)

appl_features = [
	'inittransactioncode_279L',
	'actualdpd_943P',
	'pmtnum_8L',
	'credamount_590A',
	'district_544M',
	'annuity_853A',
	'cancelreason_3545846M',
	'downpmt_134A',
	'creationdate_885D',
	'status_219L',
	'mainoccupationinc_437A',
	'rejectreason_755M',
	'rejectreasonclient_4145042M',
	'education_1138M',
	'postype_4733339M',
	'credtype_587L',
	'firstnonzeroinstldate_307D',
	'tenor_203L',
	'profession_152M',
	'credacc_credlmt_575A',
	'isbidproduct_390L']

credit_a1_features = [
	'dpdmaxdateyear_596T',
	'numberofcontrsvalue_358L',
	'numberofoverdueinstlmax_1039L',
	'classificationofcontr_13M',
	'dateofcredend_289D',
	'dateofcredstart_739D',
	'totaloutstanddebtvalue_668A',
	'totaldebtoverduevalue_178A',
	'numberofoverdueinstls_725L',
	'subjectrole_182M',
	'monthlyinstlamount_332A',
	'financialinstitution_382M',
	'description_351M',
	'financialinstitution_591M',
	'overdueamountmaxdateyear_2T',
	'overdueamount_659A',
	'contractst_545M',
	'contractst_964M',
	'purposeofcred_426M',
	'dpdmaxdatemonth_89T',
	'totaldebtoverduevalue_718A',
	'debtoverdue_47A',
	'totaloutstanddebtvalue_39A',
	'purposeofcred_874M',
	'debtoutstand_525A',
	'overdueamountmax2_14A',
	'dpdmax_139P',
	'numberofcontrsvalue_258L',
	'overdueamountmax_155A',
	'overdueamountmaxdatemonth_365T',
	'subjectrole_93M',
	'classificationofcontr_400M',
	'lastupdate_1112D']

static0_features = [
	'annuitynextmonth_57A',
	'interestrate_311L',
	'inittransactioncode_186L',
	'numcontrs3months_479L',
	'cntpmts24_3658933L',
	'lastrejectcommodtypec_5251769M',
	'lastapprcommoditycat_1041M',
	'lastapprdate_640D',
	'pctinstlsallpaidlate6d_3546844L',
	'applications30d_658L',
	'numinstpaidearly3d_3546850L',
	'posfpd30lastmonth_3976960P',
	'mobilephncnt_593L',
	'clientscnt_100L',
	'lastst_736L',
	'mastercontrelectronic_519L',
	'maxdebt4_972A',
	'currdebtcredtyperange_828A',
	'numinstregularpaid_973L',
	'applicationscnt_464L',
	'numactivecredschannel_414L',
	'daysoverduetolerancedd_3976961L',
	'maxdpdlast6m_474P',
	'firstdatedue_489D',
	'lastrejectcommoditycat_161M',
	'cntincpaycont9m_3716944L',
	'pctinstlsallpaidearl3d_427L',
	'disbursementtype_67L',
	'lastapprcommoditytypec_5251766M',
	'twobodfilling_608L',
	'applicationscnt_1086L',
	'numincomingpmts_3546848L',
	'pctinstlsallpaidlate4d_3546849L',
	'maxdpdfrom6mto36m_3546853P',
	'clientscnt12m_3712952L',
	'clientscnt_1071L',
	'mastercontrexist_109L',
	'deferredmnthsnum_166L',
	'pmtnum_254L',
	'eir_270L',
	'applicationcnt_361L',
	'clientscnt6m_3712949L',
	'maxdpdtolerance_374P',
	'actualdpdtolerance_344P',
	'clientscnt_493L',
	'numinstpaidearly5d_1087L',
	'downpmt_116A',
	'totaldebt_9A',
	'currdebt_22A',
	'clientscnt_157L',
	'pctinstlsallpaidlat10d_839L',
	'lastrejectreason_759M',
	'maininc_215A',
	'clientscnt_533L',
	'price_1097A',
	'annuity_780A',
	'lastactivateddate_801D',
	'homephncnt_628L',
	'numinstpaidlate1d_3546852L',
	'disbursedcredamount_1113A',
	'numinstunpaidmax_3546851L',
	'posfpd10lastmonth_333P',
	'numinstlswithdpd10_728L',
	'sumoutstandtotal_3546847A',
	'numnotactivated_1143L',
	'numactiverelcontr_750L',
	'numinstls_657L',
	'numinstlsallpaid_934L',
	'numactivecreds_622L',
	'clientscnt_304L',
	'applicationscnt_629L',
	'numinstpaidearly_338L',
	'paytype_783L',
	'previouscontdistrict_112M',
	'clientscnt_887L',
	'sellerplacecnt_915L',
	'lastapplicationdate_877D',
	'numinstlswithoutdpd_562L',
	'clientscnt_1022L',
	'avgdpdtolclosure24_3658938P',
	'credtype_322L',
	'lastcancelreason_561M',
	'opencred_647L',
	'clientscnt_257L',
	'clientscnt_946L',
	'lastrejectreasonclient_4145040M',
	'totalsettled_863A',
	'paytype1st_925L',
	'sellerplacescnt_216L',
	'clientscnt_360L',
	'monthsannuity_845L',
	'maxannuity_159A',
	'maxdpdlast12m_727P',
	'posfstqpd30lastmonth_3976962P',
	'isbidproduct_1095L',
	'clientscnt_1130L',
	'applicationscnt_867L',
	'numinsttopaygr_769L',
	'commnoinclast6m_3546845L',
	'maxdpdlast24m_143P',
	'pctinstlsallpaidlate1d_3546856L',
	'lastapprcredamount_781A',
	'numpmtchanneldd_318L',
	'numinstlallpaidearly3d_817L',
	'credamount_770A',
	'maxdpdlast9m_1059P',
	'clientscnt3m_3712950L',
	'maxdpdlast3m_392P',
	'numrejects9m_859L']

rest_of_features = [
	'conts_role_79M',
	'days360_512L',
	'contaddr_district_15M',
	'birth_259D',
	'secondquarter_766L',
	'numberofqueries_373L',
	'registaddr_zipcode_184M',
	'role_1084L',
	'days30_165L',
	'personindex_1023L',
	'incometype_1044T',
	'firstquarter_103L',
	'dateofbirth_337D',
	'empls_employer_name_740M',
	'addres_district_368M',
	'education_1103M',
	'education_88M',
	'empls_economicalst_849M',
	'fourthquarter_440L',
	'mainoccupationinc_384A',
	'contaddr_matchlist_1032L',
	'contaddr_zipcode_807M',
	'days90_310L',
	'registaddr_district_1083M',
	'empladdr_zipcode_114M',
	'type_25L',
	'persontype_1072L',
	'language1_981M',
	'empladdr_district_926M',
	'description_5085714M',
	'contaddr_smempladdr_334L',
	'maritalst_893M',
	'days120_123L',
	'education_927M',
	'safeguarantyflag_411L',
	'addres_zip_823M',
	'case_id',
	'sex_738L',
	'persontype_792L',
	'thirdquarter_1082L',
	'days180_256L',
	'maritalst_385M']

base = pd.read_csv(files_path + base_file[0], nrows=nrows)
base = concat_and_merge(base, applprev_files, files_path, nrows, appl_features)
base = concat_and_merge(base, credit_a1_files, files_path, nrows, credit_a1_features)
base = concat_and_merge(base, static0_files, files_path, nrows, static0_features)

for file in rest_of_files:
	data = pd.read_csv(files_path + file, nrows=nrows, low_memory=False)
	data.drop_duplicates(subset=['case_id'], keep='first', inplace=True)
	base = base.merge(data, on='case_id', how='left')
	base = merge_duplicate_group_cols(base)

del data

all_features = ['case_id', 'target', 'num_group1', 'date_decision'] + appl_features+credit_a1_features+static0_features+rest_of_features

base = base.drop(columns=base.columns.difference(all_features))

columns_to_fit = base.columns.drop(['case_id', 'target', 'num_group1', 'date_decision']).tolist()

object_cols = base[columns_to_fit].select_dtypes(include=['object']).columns
object_cols = object_cols.tolist()

encoders = dict()
for col in object_cols:
	le = LabelEncoder()
	base[col] = le.fit_transform(base[col])
	encoders[col] = le

for model_name in ['xgb', 'lightgbm']:
	results = list()
	if model_name == 'xgb':
		param_grid = {
			'n_estimators': [2000],
			'learning_rate': [0.05],
			'scale_pos_weight': [1, 30],
			'max_depth': [0, 5, 10],
			'device': ['cuda']
		}
		Classifier = XGBClassifier
	elif model_name == 'lightgbm':
		param_grid = {
			'n_estimators': [2000],
			'learning_rate': [0.05],
			'scale_pos_weight': [1, 5, 10, 20, 30, 50],
			'max_depth': [-1, 5, 10],
		}
		Classifier = LGBMClassifier
	else:
		break

	grid = ParameterGrid(param_grid)

	for params in tqdm(grid):
		model = Classifier(**params)
		model.fit(base[columns_to_fit], base['target'])
		tn, fp, fn, tp = confusion_matrix(base['target'], model.predict(base[columns_to_fit])).ravel()
		feature_importance_dict = {feature: importance for feature, importance in zip(columns_to_fit, model.feature_importances_)}
		feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
		results.append({
			'model': model_name,
			'n_estimators': params['n_estimators'],
			'learning_rate': params['learning_rate'],
			'scale_pos_weight': params['scale_pos_weight'],
			'max_depth': params['max_depth'],
			'importances': feature_importance_dict,
			'tp': tp,
			'tn': tn,
			'fp': fp,
			'fn': fn})
		break
	break
results_df = pd.DataFrame(results)

