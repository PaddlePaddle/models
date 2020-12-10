export CUDA_VISIBLE_DEVICES=0
python test.py --data_dir dataset --conf_path output/kitti_3d_multi_warmup/conf.pkl --weights_path output/kitti_3d_multi_warmup/epoch1.pdparams
