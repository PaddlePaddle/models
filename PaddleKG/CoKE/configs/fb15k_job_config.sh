
TASK=fb15k
NUM_VOCAB=16396  #NUM_VOCAB and NUM_RELATIONS must be consistent with vocab.txt file 
NUM_RELATIONS=1345

# training hyper-paramters
BATCH_SIZE=512
LEARNING_RATE=5e-4
EPOCH=300
SOFT_LABEL=0.8
SKIP_STEPS=1000
MAX_SEQ_LEN=3
HIDDEN_DROPOUT_PROB=0.1
ATTENTION_PROBS_DROPOUT_PROB=0.1

# file paths for training and evaluation 
DATA="./data"
OUTPUT="./output_${TASK}"
TRAIN_FILE="$DATA/${TASK}/train.coke.txt"
VALID_FILE="$DATA/${TASK}/valid.coke.txt"
TEST_FILE="$DATA/${TASK}/test.coke.txt"
VOCAB_PATH="$DATA/${TASK}/vocab.txt"
TRUE_TRIPLE_PATH="${DATA}/${TASK}/all.txt"
CHECKPOINTS="$OUTPUT/models"
INIT_CHECKPOINTS=$CHECKPOINTS
LOG_FILE="$OUTPUT/train.log"
LOG_EVAL_FILE="$OUTPUT/test.log"

# transformer net config, the follwoing are default configs for all tasks
HIDDEN_SIZE=256
NUM_HIDDEN_LAYERS=6
NUM_ATTENTION_HEADS=4
MAX_POSITION_EMBEDDINS=3
