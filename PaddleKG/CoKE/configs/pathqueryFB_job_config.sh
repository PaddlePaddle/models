TASK=pathqueryFB
NUM_VOCAB=75169  #NUM_VOCAB and NUM_RELATIONS must be consistent with vocab.txt file 
NUM_RELATIONS=26

# training hyper-paramters
BATCH_SIZE=2048
LEARNING_RATE=3e-4
EPOCH=10
SOFT_LABEL=1.0
SKIP_STEPS=200
MAX_SEQ_LEN=7
HIDDEN_DROPOUT_PROB=0.1
ATTENTION_PROBS_DROPOUT_PROB=0.1

# file paths for training and evaluation 
DATA="./data"
OUTPUT="./output_$TASK"
TRAIN_FILE="$DATA/${TASK}/train.coke.txt"
VALID_FILE="$DATA/${TASK}/valid.coke.txt"
TEST_FILE="$DATA/${TASK}/test.coke.txt"
VOCAB_PATH="$DATA/${TASK}/vocab.txt"
TRUE_TRIPLE_PATH="${DATA}/${TASK}/all.txt"

SEN_CANDLI_PATH="$DATA/${TASK}/sen_candli.txt"
TRIVAL_SEN_PATH="$DATA/${TASK}/trivial_sen.txt"

CHECKPOINTS="$OUTPUT/models"
LOG_FILE="$OUTPUT/train.log"
LOG_EVAL_FILE="$OUTPUT/test.log"

# transformer net config, the follwoing are default configs for all tasks
HIDDEN_SIZE=256
NUM_HIDDEN_LAYERS=6
NUM_ATTENTION_HEADS=4
MAX_POSITION_EMBEDDINS=7
