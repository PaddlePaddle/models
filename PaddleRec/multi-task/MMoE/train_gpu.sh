CUDA_VISIBLE_DEVICES=0 python mmoe_train.py --use_gpu 1\
                                            --train_data_path 'train_data'\
                                            --test_data_path 'test_data'\
<<<<<<< HEAD
                                            --model_dir 'model_dir'\
=======
>>>>>>> 282e48904fbd6168835966b4e0c7851c82d46e23
                                            --feature_size 499\
                                            --batch_size 32\
                                            --expert_num 8\
                                            --gate_num 2\
                                            --expert_size 16\
                                            --tower_size 8\
                                            --epochs 100
