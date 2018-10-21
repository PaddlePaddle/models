source ~/mapingshuo/.bash_mapingshuo_fluid

export CUDA_VISIBLE_DEVICES=1

fluid train_and_evaluate.py  \
    --model_name=cdssmNet  \
    --config=cdssm_base

#fluid train_and_evaluate.py \
#    --model_name=DecAttNet  \
#    --config=decatt_glove

#fluid train_and_evaluate.py \
#    --model_name=DecAttNet  \
#    --config=decatt_word


#fluid train_and_evaluate.py  \
#    --model_name=ESIMNet  \
#    --config=esim_seq
