mkdir train_data
mkdir test_data
mkdir data
train_path="data/adult.data"
test_path="data/adult.test"
train_data_path="train_data/train_data.csv"
test_data_path="test_data/test_data.csv"

pip install -r requirements.txt

wget -P data/ https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
wget -P data/ https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test

python data_preparation.py --train_path ${train_path} \
                           --test_path ${test_path} \
                           --train_data_path ${train_data_path}\
                           --test_data_path ${test_data_path}
