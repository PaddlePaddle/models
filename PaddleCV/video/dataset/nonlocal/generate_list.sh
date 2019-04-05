# Download txt name
TRAINLIST_DOWNLOAD="kinetics-400_train.csv"

# path of the train and valid data
TRAIN_DIR="/home/sungaofeng/docker/dockermount/data/compress/train_256"
VALID_DIR="/home/sungaofeng/docker/dockermount/data/compress/val_256"

python generate_filelist.py $TRAINLIST_DOWNLOAD $TRAIN_DIR $VALID_DIR trainlist.txt vallist.txt

# generate test list
python generate_testlist_multicrop.py

