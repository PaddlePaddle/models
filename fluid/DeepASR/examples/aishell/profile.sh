export CUDA_VISIBLE_DEVICES=2,3,4,5
python -u ../../tools/profile.py --feature_lst data/train_feature.lst \
                   --label_lst data/train_label.lst \
                   --mean_var data/aishell/global_mean_var \
                   --parallel \
                   --frame_dim 2640  \
                   --class_num 101  \
