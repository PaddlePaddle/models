CUDA_VISIBLE_DEVICES=0 python infer.py --batch_size 40 \
                                        --use_gpu 1 \
                                        --test_epoch 39 \
                                        --test_data_path 'test_data/test_data.csv' \
                                        --model_dir 'model_dir' \
                                        --hidden1_units 75 \
                                        --hidden2_units 50 \
                                        --hidden3_units 25
                
                