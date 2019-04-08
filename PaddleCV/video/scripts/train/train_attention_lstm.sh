export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py --model_name="AttentionLSTM" --config=./configs/attention_lstm.txt --epoch_num=10 \
                --valid_interval=1 --log_interval=10
