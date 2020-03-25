mkdir data/data24913/train_data
mkdir data/data24913/test_data
mkdir data/data24913/validation_data

train_path="data/data24913/census-income.data"
test_path="data/data24913/census-income.test"
train_data_path="data/data24913/train_data/"
test_data_path="data/data24913/test_data/"
validation_data_path="data/data24913/validation_data/"

python data_preparation.py --train_path ${train_path} \
                           --test_path ${test_path} \
                           --train_data_path ${train_data_path}\
                           --test_data_path ${test_data_path}\
                           --validation_data_path ${validation_data_path}