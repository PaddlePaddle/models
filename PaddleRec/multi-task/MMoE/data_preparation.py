import pandas as pd
import numpy as np
import paddle.fluid as fluid
from args import *


def fun1(x):
    if x == ' 50000+.':
        return 1
    else:
        return 0


def fun2(x):
    if x == ' Never married':
        return 1
    else:
        return 0


def data_preparation(train_path, test_path, train_data_path, test_data_path,
                     validation_data_path):
    # The column names are from
    # https://www2.1010data.com/documentationcenter/prod/Tutorials/MachineLearningExamples/CensusIncomeDataSet.html
    column_names = [
        'age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education',
        'wage_per_hour', 'hs_college', 'marital_stat', 'major_ind_code',
        'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
        'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses',
        'stock_dividends', 'tax_filer_stat', 'region_prev_res',
        'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ', 'instance_weight',
        'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same',
        'mig_prev_sunbelt', 'num_emp', 'fam_under_18', 'country_father',
        'country_mother', 'country_self', 'citizenship', 'own_or_self',
        'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k'
    ]

    # Load the dataset in Pandas
    train_df = pd.read_csv(
        train_path,
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names)
    other_df = pd.read_csv(
        test_path,
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names)

    # First group of tasks according to the paper
    label_columns = ['income_50k', 'marital_stat']

    # One-hot encoding categorical columns
    categorical_columns = [
        'class_worker', 'det_ind_code', 'det_occ_code', 'education',
        'hs_college', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin',
        'sex', 'union_member', 'unemp_reason', 'full_or_part_emp',
        'tax_filer_stat', 'region_prev_res', 'state_prev_res',
        'det_hh_fam_stat', 'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg',
        'mig_move_reg', 'mig_same', 'mig_prev_sunbelt', 'fam_under_18',
        'country_father', 'country_mother', 'country_self', 'citizenship',
        'vet_question'
    ]
    train_raw_labels = train_df[label_columns]
    other_raw_labels = other_df[label_columns]
    transformed_train = pd.get_dummies(train_df, columns=categorical_columns)
    transformed_other = pd.get_dummies(other_df, columns=categorical_columns)

    # Filling the missing column in the other set
    transformed_other[
        'det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0
    # get label
    transformed_train['income_50k'] = transformed_train['income_50k'].apply(
        lambda x: fun1(x))
    transformed_train['marital_stat'] = transformed_train['marital_stat'].apply(
        lambda x: fun2(x))
    transformed_other['income_50k'] = transformed_other['income_50k'].apply(
        lambda x: fun1(x))
    transformed_other['marital_stat'] = transformed_other['marital_stat'].apply(
        lambda x: fun2(x))
    # Split the other dataset into 1:1 validation to test according to the paper
    validation_indices = transformed_other.sample(
        frac=0.5, replace=False, random_state=1).index
    test_indices = list(set(transformed_other.index) - set(validation_indices))
    validation_data = transformed_other.iloc[validation_indices]
    test_data = transformed_other.iloc[test_indices]

    cols = transformed_train.columns.tolist()
    cols.insert(0, cols.pop(cols.index('income_50k')))
    cols.insert(0, cols.pop(cols.index('marital_stat')))
    transformed_train = transformed_train[cols]
    test_data = test_data[cols]
    validation_data = validation_data[cols]

    print(transformed_train.shape, transformed_other.shape,
          validation_data.shape, test_data.shape)
    transformed_train.to_csv(train_data_path + 'train_data.csv', index=False)
    test_data.to_csv(test_data_path + 'test_data.csv', index=False)


args = data_preparation_args()
data_preparation(args.train_path, args.test_path, args.train_data_path,
                 args.test_data_path, args.validation_data_path)
