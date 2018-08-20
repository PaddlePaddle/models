export ce_mode=1
python ctc_train.py --batch_size=32 --total_step=1 --eval_period=1 --log_period=1 --use_gpu=True 1> ./tmp.log
cat tmp.log | python _ce.py
rm tmp.log
