export ce_mode=1
rm -f *_factor.txt
python train.py --batch_size=32 --total_step=100 --eval_period=100 --log_period=100 --use_gpu=True 1> ./tmp.log
cat tmp.log | python _ce.py
rm tmp.log
