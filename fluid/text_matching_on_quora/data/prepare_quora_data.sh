

# Please download the Quora dataset firstly from https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing
# to the ROOT_DIR: $HOME/.cache/paddle/dataset

DATA_DIR=$HOME/.cache/paddle/dataset
wget --directory-prefix=$DATA_DIR http://nlp.stanford.edu/data/glove.840B.300d.zip

unzip $DATA_DIR/glove.840B.300d.zip

# The finally dataset dir should be like

# $HOME/.cache/paddle/dataset
# |- Quora_question_pair_partition
#     |- train.tsv
#     |- test.tsv
#     |- dev.tsv
#     |- readme.txt
#     |- wordvec.txt
# |- glove.840B.300d.txt
