python infer.py --model-name="AttentionLSTM" --config=./configs/attention_lstm.txt \
                --filelist=./data/youtube8m/infer.list \
                --weights=./checkpoints/AttentionLSTM_epoch0 \
                --save-dir="./save"
