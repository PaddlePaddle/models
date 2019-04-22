export CPU_NUM=1
export FLAGS_eager_delete_tensor_gb=0.0

#train on ubuntu 
python -u main.py \
  --do_train True \
  --data_path ./data/ubuntu/data_small.pkl \
  --save_path ./model_files_cpu/ubuntu \
  --use_pyreader \
  --stack_num 2 \
  --vocab_size 434512 \
  --_EOS_ 28270 \
  --batch_size 32

#test on ubuntu
python -u main.py \
  --do_test True \
  --data_path ./data/ubuntu/data_small.pkl \
  --save_path ./model_files_cpu/ubuntu/step_31 \
  --model_path ./model_files_cpu/ubuntu/step_31 \
  --stack_num 2 \
  --vocab_size 434512 \
  --_EOS_ 28270 \
  --batch_size 40

#train on douban
python -u main.py \
  --do_train True \
  --data_path ./data/douban/data_small.pkl \
  --save_path ./model_files_cpu/douban \
  --use_pyreader \
  --stack_num 2 \
  --vocab_size 172130 \
  --_EOS_ 1 \
  --channel1_num 16 \
  --batch_size 32

#test on douban
python -u main.py \
  --do_test True \
  --ext_eval \
  --data_path ./data/douban/data_small.pkl \
  --save_path ./model_files_cpu/douban/step_31 \
  --model_path ./model_files_cpu/douban/step_31 \
  --stack_num 2 \
  --vocab_size 172130 \
  --_EOS_ 1 \
  --channel1_num 16 \
  --batch_size 40
