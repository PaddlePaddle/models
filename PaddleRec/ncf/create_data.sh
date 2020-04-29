mkdir Data
pip install -r requirements.txt
wget -P Data https://paddlerec.bj.bcebos.com/ncf/Data.zip
unzip Data/Data.zip -d Data/
python get_train_data.py --num_neg 4 \
                --train_data_path "Data/train_data.csv"