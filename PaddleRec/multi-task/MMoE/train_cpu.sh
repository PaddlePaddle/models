python mmoe_train.py --use_gpu 0\
                    --train_data_path 'train_data'\
                    --test_data_path 'test_data'\
                    --feature_size 499\
                    --batch_size 32\
                    --expert_num 8\
                    --gate_num 2\
                    --expert_size 16\
                    --tower_size 8\
                    --epochs 100
