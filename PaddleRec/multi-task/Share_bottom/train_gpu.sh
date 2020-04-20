CUDA_VISIBLE_DEVICES=0 python share_bottom.py --use_gpu 1 \
                       --epochs 100 \
                       --train_data_path 'train_data' \
                       --test_data_path 'test_data' \
                       --train_data_path '.train_data' \
                       --test_data_path '.test_data' \
                       --model_dir 'model_dir' \
                       --batch_size 32 \
                       --feature_size 499 \
                       --bottom_size 117 \
                       --tower_nums 2 \
                       --tower_size 8 



