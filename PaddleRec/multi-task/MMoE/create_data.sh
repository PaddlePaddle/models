mkdir train_data
mkdir test_data
mkdir data
train_path="data/census-income.data"
test_path="data/census-income.test"
train_data_path="train_data/"
test_data_path="test_data/"

pip install -r requirements.txt

<<<<<<< HEAD
#wget -P data/ https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census.tar.gz
#tar -zxvf data/census.tar.gz -C data/
=======
wget -P data/ https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census.tar.gz
tar -zxvf data/census.tar.gz -C data/
>>>>>>> 6c87d487199034dde7d0d5dde4ef324fe5d273f1

python data_preparation.py --train_path ${train_path} \
                           --test_path ${test_path} \
                           --train_data_path ${train_data_path}\
                           --test_data_path ${test_data_path}
