CUDA_VISIBLE_DEVICES=0 python train.py --use_gpu 1 \
                                        --MLP 1 \
                                        --epochs 20 \
                                        --batch_size 256 \
                                        --num_factors 8 \
                                        --num_neg 4 \
                                        --lr 0.001 \
                                        --model_dir 'mlp_model_dir' 