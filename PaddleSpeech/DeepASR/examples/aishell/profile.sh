export CUDA_VISIBLE_DEVICES=0
python -u ../../tools/profile.py --feature_lst data/train_feature.lst \
                   --label_lst data/train_label.lst \
                   --mean_var data/global_mean_var \
                   --frame_dim 80  \
                   --class_num 3040  \
                   --batch_size 16
