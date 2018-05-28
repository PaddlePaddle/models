export CUDA_VISIBLE_DEVICES=2,3,4,5
python -u ../../train.py --train_feature_lst data/train_feature.lst \
                   --train_label_lst data/train_label.lst \
                   --val_feature_lst data/val_feature.lst \
                   --val_label_lst data/val_label.lst \
                   --mean_var data/aishell/global_mean_var \
                   --checkpoints checkpoints \
                   --frame_dim 2640  \
                   --class_num 101  \
                   --infer_models '' \
                   --batch_size 128 \
                   --learning_rate 0.00016 \
                   --parallel
