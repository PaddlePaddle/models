export CUDA_VISIBLE_DEVICES=0
cd data 
sh download_data.sh
cd ..

python  train.py \
    --data_path data/simple-examples/data/ \
    --model_type small \
    --use_gpu True \
    --rnn_model static \
    --enable_ce | python _ce.py

