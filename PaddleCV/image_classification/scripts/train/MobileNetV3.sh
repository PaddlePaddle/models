#Training details
#GPU: NVIDIA® Tesla® V100 4cards 120epochs 55h
#export CUDA_VISIBLE_DEVICES=0,1
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0


python quantization/train.py \
       --model=MobileNetV3_large_x1_0 \
       --batch_size=256 \
       --model_save_dir=output/Mobilenetv3_large_x1_0_78_16_quant \
       --lr_strategy=piecewise_decay \
       --num_epochs=30 \
       --step_epoch=20 \
       --lr=0.0001 \
       --l2_decay=3e-5 \
       --use_label_smoothing True \
       --pretrained_model ./pretrain_model/MobileNetV3_large_x1_0_78.16_pretrained/ \
     #  > Mobilenetv3_large_x1_0_78_quant.log 2>&1 &
