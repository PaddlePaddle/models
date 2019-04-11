export CUDA_VISIBLE_DEVICES=3
export FLAGS_eager_delete_tensor_gb=0.0

#train on ubuntu 
python -u main.py \
  --do_train True \
  --use_cuda \
  --data_path ./data/ubuntu/data_small.pkl \
  --save_path ./model_files/ubuntu \
  --use_pyreader \
  --vocab_size 434512 \
  --_EOS_ 28270 \
  --batch_size 32

#test on ubuntu
python -u main.py \
  --do_test True \
  --use_cuda \
  --data_path ./data/ubuntu/data_small.pkl \
  --save_path ./model_files/ubuntu/step_31 \
  --model_path ./model_files/ubuntu/step_31 \
  --vocab_size 434512 \
  --_EOS_ 28270 \
  --batch_size 100

#train on douban
python -u main.py \
  --do_train True \
  --use_cuda \
  --data_path ./data/douban/data_small.pkl \
  --save_path ./model_files/douban \
  --use_pyreader \
  --vocab_size 172130 \
  --_EOS_ 1 \
  --channel1_num 16 \
  --batch_size 32

#test on douban
python -u main.py \
  --do_test True \
  --use_cuda \
  --ext_eval \
  --data_path ./data/douban/data_small.pkl \
  --save_path ./model_files/douban/step_31 \
  --model_path ./model_files/douban/step_31 \
  --vocab_size 172130 \
  --_EOS_ 1 \
  --channel1_num 16 \
  --batch_size 32
