export ce_mode=1
rm *factor.txt
python ctc_train.py --batch_size=32 --total_step=30000 --eval_period=30000 --log_period=30000 --use_gpu=True 1> ./tmp.log
cat tmp.log | python _ce.py
rm tmp.log
