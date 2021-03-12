d=$1
python train.py --test_fold=1 --device=$d
python train.py --test_fold=2 --device=$d
python train.py --test_fold=3 --device=$d
python train.py --test_fold=4 --device=$d
python train.py --test_fold=5 --device=$d
