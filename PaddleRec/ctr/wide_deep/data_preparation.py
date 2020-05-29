import os
import io
import args
import pandas as pd
from sklearn import  preprocessing

def _clean_file(source_path,target_path):
    """makes changes to match the CSV format."""
    with io.open(source_path, 'r') as temp_eval_file:
        with io.open(target_path, 'w') as eval_file:
            for line in temp_eval_file:
                line = line.strip()
                line = line.replace(', ', ',')
                if not line or ',' not in line:
                    continue
                if line[-1] == '.':
                    line = line[:-1]
                line += '\n'
                eval_file.write(line)
                    
def build_model_columns(train_data_path, test_data_path):
    # The column names are from
    # https://www2.1010data.com/documentationcenter/prod/Tutorials/MachineLearningExamples/CensusIncomeDataSet.html
    column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
    ]

    # Load the dataset in Pandas
    train_df = pd.read_csv(
        train_data_path,
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names)
    test_df = pd.read_csv(
        test_data_path,
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names)

    # First group of tasks according to the paper
    #label_columns = ['income_50k', 'marital_stat']
    categorical_columns = ['education','marital_status','relationship','workclass','occupation']
    for col in categorical_columns:
        label_train = preprocessing.LabelEncoder()
        train_df[col]= label_train.fit_transform(train_df[col])
        label_test = preprocessing.LabelEncoder()
        test_df[col]= label_test.fit_transform(test_df[col])
    
    bins = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]  
    train_df['age_buckets'] = pd.cut(train_df['age'].values.tolist(), bins,labels=False)
    test_df['age_buckets'] = pd.cut(test_df['age'].values.tolist(), bins,labels=False)
    
    base_columns = ['education', 'marital_status', 'relationship', 'workclass', 'occupation', 'age_buckets']
    
    train_df['education_occupation'] = train_df['education'].astype(str) + '_' + train_df['occupation'].astype(str)    
    test_df['education_occupation'] = test_df['education'].astype(str) + '_' + test_df['occupation'].astype(str)
    train_df['age_buckets_education_occupation'] = train_df['age_buckets'].astype(str) + '_' + train_df['education'].astype(str) + '_' + train_df['occupation'].astype(str)
    test_df['age_buckets_education_occupation'] = test_df['age_buckets'].astype(str) + '_' + test_df['education'].astype(str) + '_' + test_df['occupation'].astype(str)
    crossed_columns = ['education_occupation','age_buckets_education_occupation']
    
    for col in crossed_columns:
        label_train = preprocessing.LabelEncoder()
        train_df[col]= label_train.fit_transform(train_df[col])
        label_test = preprocessing.LabelEncoder()
        test_df[col]= label_test.fit_transform(test_df[col])
        
    wide_columns = base_columns + crossed_columns
    
    train_df_temp = pd.get_dummies(train_df[categorical_columns],columns=categorical_columns)
    test_df_temp = pd.get_dummies(test_df[categorical_columns], columns=categorical_columns)
    train_df = train_df.join(train_df_temp)
    test_df = test_df.join(test_df_temp)
    
    deep_columns = list(train_df_temp.columns)+ ['age','education_num','capital_gain','capital_loss','hours_per_week']
    
    train_df['label'] = train_df['income_bracket'].apply(lambda x : 1 if x == '>50K' else 0)
    test_df['label'] = test_df['income_bracket'].apply(lambda x : 1 if x == '>50K' else 0)
    
    with open('train_data/columns.txt','w') as f:
        write_str = str(len(wide_columns)) + '\n' + str(len(deep_columns)) + '\n'
        f.write(write_str)
        f.close()
    with open('test_data/columns.txt','w') as f:
        write_str = str(len(wide_columns)) + '\n' + str(len(deep_columns)) + '\n'
        f.write(write_str)
        f.close()
    
    train_df[wide_columns + deep_columns + ['label']].fillna(0).to_csv(train_data_path,index=False)
    test_df[wide_columns + deep_columns + ['label']].fillna(0).to_csv(test_data_path,index=False)


def clean_file(train_path, test_path, train_data_path, test_data_path):
    _clean_file(train_path, train_data_path)
    _clean_file(test_path, test_data_path)

if __name__ == '__main__':
    args = args.parse_args()
    clean_file(args.train_path, args.test_path, args.train_data_path, args.test_data_path)  
    build_model_columns(args.train_data_path, args.test_data_path)
    
    
    
    
        