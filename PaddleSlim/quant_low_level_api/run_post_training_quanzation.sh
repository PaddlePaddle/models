export CUDA_VISIBLE_DEVICES=0

root_url="https://paddle-inference-dist.bj.bcebos.com/int8"
mobilenetv1="mobilenetv1_fp32_model"
samples="samples_100"
if [ ! -d ${mobilenetv1} ]; then
    wget ${root_url}/${mobilenetv1}.tgz
    tar zxf ${mobilenetv1}.tgz
fi
if [ ! -d ${samples} ]; then
    wget ${root_url}/${samples}.tgz
    tar zxf ${samples}.tgz
fi

python post_training_quantization.py \
    --model_dir=${mobilenetv1} \
    --data_path=${samples} \
    --save_model_path="mobilenetv1_int8_model" \
    --algo="KL" \
    --is_full_quantize=False \
    --batch_size=10 \
    --batch_nums=10 \
    --use_gpu=True \
