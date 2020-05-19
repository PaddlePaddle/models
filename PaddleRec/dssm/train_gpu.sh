CUDA_VISIBLE_DEVICES=0 python dssm.py --use_gpu 1 \
                                   --batch_size 16 \
                                   --TRIGRAM_D 1000 \
                                   --L1_N 300 \
                                   --L2_N 300 \
                                   --L3_N 128 \
                                   --Neg 4 \
                                   --base_lr 0.01 \
                                   --model_dir ./model_dir