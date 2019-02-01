1. download kinetics-400_train.csv and kinetics-400_val.csv
2. ffmpeg is required to decode mp4
3. transfer mp4 video to pkl file, with each pkl stores [video_id, images, label]
   python generate_label.py kinetics-400_train.csv kinetics400_label.txt # generate label file
   python video2pkl.py kinetics-400_train.csv $Source_dir $Target_dir $NUM_THREADS
