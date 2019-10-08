##Training details
#GPU: NVIDIA® Tesla® V100 4cards 200epochs 141h
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fraction_of_gpu_memory_to_use=0.98

#ResNet50_vc
python train.py \
  	    --model=ResNet50_vc \
	    --batch_size=256 \
            --total_images=1281167 \
            --image_shape=3,224,224 \
            --class_dim=1000 \
            --lr_strategy=cosine_decay \
            --lr=0.1 \
            --num_epochs=200 \
            --model_save_dir=output/ \
            --l2_decay=1e-4 \
