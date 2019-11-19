export CUDA_VISIBLE_DEVICES=0

root_url="https://paddle-inference-dist.bj.bcebos.com/int8"
mobilenetv1="mobilenetv1_fp32_model.zip"
if [ ! -f ${mobilenetv1} ]; then
    wget ${root_url}/${mobilenetv1}
    unzip ${mobilenetv1}
fi

python post_training_quantization.py \
    --model_dir="mobilenetv1_fp32_model" \
    --data_path="/dataset/ILSVRC2012/" \
    --save_model_path="mobilenetv1_int8_model" \
    --algo="KL" \
    --is_full_quantize=True \
    --batch_size=10 \
    --batch_nums=10 \
    --use_gpu=True \
