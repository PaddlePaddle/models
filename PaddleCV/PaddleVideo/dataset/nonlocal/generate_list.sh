# Download txt name
TRAINLIST_DOWNLOAD="kinetics-400_train.csv"

# path of the train and valid data
TRAIN_DIR=YOUR_TRAIN_DATA_DIR # replace this with your train data dir
VALID_DIR=YOUR_VALID_DATA_DIR # replace this with your valid data dir

python generate_filelist.py $TRAINLIST_DOWNLOAD $TRAIN_DIR $VALID_DIR trainlist.txt vallist.txt

# generate test list
python generate_testlist_multicrop.py

