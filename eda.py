import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def get_file_names(file_names, keyword):
    return [x for x in file_names if keyword in x]


def concat_files(path_list, base_path, rows):
    data = pd.DataFrame()
    for file in path_list:
        data = pd.concat([data, pd.read_csv(base_path + file, nrows=rows, low_memory=False)], axis=0)

    #data.drop_duplicates(subset=['case_id'], keep='first', inplace=True)
    return data


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

applprev = concat_files(applprev_files, files_path, nrows)


def get_null_analysis(col, data):
    print(f'missing ratio on {col}:', data[col].isnull().sum() / len(data))
    value_counts = data[col].value_counts().reset_index()

    data['isnull'] = data[col].isnull().astype(int)
    cases = data[['case_id', 'isnull']].copy()

    return value_counts, cases

value_counts, cases = get_null_analysis(col = 'actualdpd_943P', data=applprev)

desc = applprev[col].describe()
desc.apply(lambda x: '{:.2f}'.format(x))

col = 'annuity_853A'
print(f'missing ratio on {col}:', applprev[col].isnull().sum() / len(applprev))
value_counts = applprev[col].value_counts().reset_index()

sns.histplot(value_counts, x=col, weights='count', bins=50)
desc = applprev[col].describe()
desc.apply(lambda x: '{:.2f}'.format(x))

# for date column
col = 'approvaldate_319D'
print(f'missing ratio on {col}:', applprev[col].isnull().sum() / len(applprev))
value_counts = applprev[col].value_counts().reset_index()

value_counts[col] = pd.to_datetime(value_counts[col])
value_counts['date_for_plot'] = pd.to_datetime(dict(year=value_counts[col].dt.year, month=value_counts[col].dt.month, day=1))
sns.barplot(value_counts['date_for_plot'].sort_values(ascending=True))

col = 'byoccupationinc_3656910L'
print(f'missing ratio on {col}:', applprev[col].isnull().sum() / len(applprev))
value_counts = applprev[col].value_counts(normalize=True).reset_index()

sns.histplot(value_counts, x='byoccupationinc_3656910L', weights='count', bins=200)
desc = applprev[col].describe()
desc.apply(lambda x: '{:.2f}'.format(x))

a = ['cancelreason_3545846M', 'childnum_21L',
       'creationdate_885D', 'credacc_actualbalance_314A',
       'credacc_credlmt_575A', 'credacc_maxhisbal_375A',
       'credacc_minhisbal_90A', 'credacc_status_367L',
       'credacc_transactions_402L', 'credamount_590A', 'credtype_587L',
       'currdebt_94A', 'dateactivated_425D', 'district_544M', 'downpmt_134A',
       'dtlastpmt_581D', 'dtlastpmtallstes_3545839D', 'education_1138M',
       'employedfrom_700D', 'familystate_726L', 'firstnonzeroinstldate_307D',
       'inittransactioncode_279L', 'isbidproduct_390L', 'isdebitcard_527L',
       'mainoccupationinc_437A', 'maxdpdtolerance_577P', 'num_group1',
       'outstandingdebt_522A', 'pmtnum_8L', 'postype_4733339M',
       'profession_152M', 'rejectreason_755M', 'rejectreasonclient_4145042M',
       'revolvingaccount_394A', 'status_219L', 'tenor_203L',
       'cacccardblochreas_147M', 'conts_type_509L', 'credacc_cards_status_52L',
       'num_group2']

col = 'cancelreason_3545846M'
print(f'missing ratio on {col}:', applprev[col].isnull().sum() / len(applprev))
value_counts = applprev[col].value_counts(normalize=True).reset_index()

ax = sns.barplot(value_counts, x=col, y='proportion')
plt.xticks(rotation=90)

col = 'childnum_21L'
print(f'missing ratio on {col}:', applprev[col].isnull().sum() / len(applprev))
value_counts = applprev[col].value_counts(normalize=True).reset_index()

col = 'creationdate_885D'
print(f'missing ratio on {col}:', applprev[col].isnull().sum() / len(applprev))
value_counts = applprev[col].value_counts().reset_index()
value_counts[col] = pd.to_datetime(value_counts[col])
value_counts['date_for_plot'] = pd.to_datetime(dict(year=value_counts[col].dt.year, month=value_counts[col].dt.month, day=1))
sns.barplot(value_counts['date_for_plot'].sort_values(ascending=True))

col = 'credacc_actualbalance_314A'
print(f'missing ratio on {col}:', applprev[col].isnull().sum() / len(applprev))
value_counts = applprev[col].value_counts().reset_index()
sns.histplot(value_counts, x=col, weights='count', bins=200)
