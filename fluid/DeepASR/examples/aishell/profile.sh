export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u ../../tools/profile.py --feature_lst data/train_feature.lst \
                   --label_lst data/train_label.lst \
                   --mean_var data/aishell/global_mean_var \
                   --parallel \
                   --frame_dim 80  \
                   --class_num 3040  \
