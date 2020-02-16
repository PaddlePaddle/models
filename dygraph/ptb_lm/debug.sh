
export CUDA_VISIBLE_DEVICES=0

DATA_PATH="data/simple-examples/data/"
MODEL_TYPE="small"

while getopts d:t: opt
do  
    case $opt in
        d)
            DATA_PATH="$OPTARG"
            ;;
        t)
            MODEL_TYPE="$OPTARG"
            ;;
        \?)
            exit;  
            ;;
    esac
done
echo "python  ptb_dy.py  --data_path $DATA_PATH  --model_type $MODEL_TYPE"
python  ptb_dy.py  --data_path $DATA_PATH  --model_type $MODEL_TYPE 
